"""Synapse retrieval engine (plan Part 5). How knowledge comes out.

Multi-strategy retrieval, a tunable ranking algorithm, scope composition, and the
``brief()`` killer feature with Redis caching.

**Semantic search runs over Neo4j's vector index via Graphiti** (`graphiti.search`
already combines cosine similarity + BM25 + graph BFS), not a separate Qdrant —
the write pipeline stores embeddings through Graphiti into Neo4j, so that is where
the vectors live. This confirms the §6C question: Qdrant is currently redundant.
See docs/architecture/retrieval.md.

All graph access is behind injected Protocols (`Searcher`, `GraphQueries`) so the
ranking, temporal filtering, scope composition, and brief assembly are unit-tested
without live services.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Protocol

from pydantic import BaseModel, Field

from synapse.config import settings
from synapse.core.schema import Scope

logger = logging.getLogger(__name__)

# Knowledge-type labels used to bucket a brief.
_CONVENTION, _DECISION, _LESSON, _PATTERN = "Convention", "Decision", "Lesson", "Pattern"


# ──────────────────────────────────────────────────────────────────────────────
# Normalized data shapes (decoupled from Graphiti types → easy to fake)
# ──────────────────────────────────────────────────────────────────────────────


class Fact(BaseModel):
    uuid: str
    fact: str
    group_id: str
    created_at: datetime | None = None
    valid_at: datetime | None = None
    invalid_at: datetime | None = None
    source_uuid: str | None = None
    target_uuid: str | None = None
    confidence: float | None = None
    score: float | None = None  # raw similarity from Graphiti/Neo4j (None if unavailable)


class NodeRow(BaseModel):
    uuid: str
    name: str
    summary: str | None = None
    labels: list[str] = Field(default_factory=list)
    group_id: str
    created_at: datetime | None = None
    attributes: dict = Field(default_factory=dict)


class RankWeights(BaseModel):
    """Tunable ranking weights (need not sum to 1; they're normalized at use)."""

    relevance: float = 0.45
    recency: float = 0.20
    confidence: float = 0.20
    connectivity: float = 0.15
    recency_half_life_days: float = 30.0


class Recalled(BaseModel):
    fact: str
    score: float
    scope: str
    uuid: str
    valid_at: datetime | None = None
    components: dict[str, float] = Field(default_factory=dict)


class Brief(BaseModel):
    project_id: str
    project_summary: str
    active_conventions: list[str]
    key_decisions: list[str]
    relevant_lessons: list[str]
    cross_project_knowledge: list[str]
    generated_at: datetime
    cached: bool = False


# ──────────────────────────────────────────────────────────────────────────────
# Injected collaborators
# ──────────────────────────────────────────────────────────────────────────────


class Searcher(Protocol):
    async def search(
        self, query: str, scopes: list[str], limit: int, center_node_uuid: str | None
    ) -> list[Fact]: ...


class GraphQueries(Protocol):
    async def nodes_by_label(self, labels: list[str], scopes: list[str], limit: int) -> list[NodeRow]: ...
    async def degrees(self, node_uuids: list[str]) -> dict[str, int]: ...


# ──────────────────────────────────────────────────────────────────────────────
# Pure functions (the ranking + temporal logic — directly unit-tested)
# ──────────────────────────────────────────────────────────────────────────────


def temporal_filter(facts: list[Fact], as_of: datetime | None) -> list[Fact]:
    """Keep only facts valid at ``as_of`` (default: now).

    A fact is valid at T when ``valid_at <= T`` and (``invalid_at`` is None or
    ``invalid_at > T``). Facts with no temporal bounds are always kept.
    """
    if as_of is None:
        as_of = datetime.now(timezone.utc)
    kept = []
    for f in facts:
        if f.valid_at is not None and f.valid_at > as_of:
            continue
        if f.invalid_at is not None and f.invalid_at <= as_of:
            continue
        kept.append(f)
    return kept


def apply_similarity_floor(facts: list[Fact], min_relevance: float | None) -> list[Fact]:
    """Drop candidates whose absolute similarity ``score`` is below ``min_relevance``.

    The floor is a guard against *confident junk*: BGE-M3 gives a high baseline
    cosine (~0.65) to unrelated text, so an off-topic query still returns a full,
    plausible-looking result set. A raw-cosine floor (~0.72–0.75) removes those
    (they top out near 0.68) while leaving genuine hits (typically ≥ 0.80) intact.

    * ``min_relevance is None`` or ``<= 0`` → no-op (floor disabled).
    * A fact with ``score is None`` (no absolute similarity available — e.g. the
      searcher couldn't compute cosine) is KEPT: we never had a signal to reject
      it on, and silently dropping everything would be worse than a soft miss.

    Applied to the *raw* similarity, not the per-result-set min-max normalized
    relevance — normalization is relative (an off-topic set still has a "best"
    result at 1.0), so only the absolute score can separate topics.
    """
    if not min_relevance or min_relevance <= 0:
        return facts
    return [f for f in facts if f.score is None or f.score >= min_relevance]


def _recency_score(when: datetime | None, now: datetime, half_life_days: float) -> float:
    if when is None:
        return 0.5
    age_days = max(0.0, (now - when).total_seconds() / 86400.0)
    return 0.5 ** (age_days / half_life_days)


def _relevance_scores(facts: list[Fact]) -> list[float]:
    """Per-fact relevance in [0, 1].

    Locked decision: if any fact carries a real similarity ``score``, min-max
    normalize those (a missing score gets a neutral 0.5); otherwise fall back to
    positional ``1 - i / max(n, 1)`` from the searcher's incoming order.
    """
    scores = [f.score for f in facts]
    present = [s for s in scores if s is not None]
    if present:
        lo, hi = min(present), max(present)
        span = hi - lo
        return [
            0.5 if s is None else (1.0 if span == 0 else (s - lo) / span)
            for s in scores
        ]
    n = max(len(facts), 1)
    return [1.0 - (i / n) for i in range(len(facts))]


def score_facts(
    facts: list[Fact],
    weights: RankWeights,
    connectivity: dict[str, float],
    now: datetime | None = None,
) -> list[tuple[Fact, float, dict[str, float]]]:
    """Combine relevance + recency + confidence + connectivity.

    Relevance uses the real similarity ``f.score`` (min-max normalized across the
    result set) when any fact carries one; otherwise it falls back to positional
    ``1 - i/n`` from the searcher's relevance order. Returns (fact, score,
    components) sorted by descending composite score.
    """
    now = now or datetime.now(timezone.utc)
    n = len(facts)
    total_w = (weights.relevance + weights.recency + weights.confidence + weights.connectivity) or 1.0
    relevances = _relevance_scores(facts)
    scored = []
    for i, f in enumerate(facts):
        relevance = relevances[i]
        recency = _recency_score(f.valid_at or f.created_at, now, weights.recency_half_life_days)
        confidence = f.confidence if f.confidence is not None else 0.5
        # Facts with no degree data (manual relate() edges, endpoints outside the batch) get 0.0,
        # not a neutral 0.5 — an unknown connection count must not outrank a measured one.
        conn = connectivity.get(f.uuid, 0.0)
        composite = (
            weights.relevance * relevance
            + weights.recency * recency
            + weights.confidence * confidence
            + weights.connectivity * conn
        ) / total_w
        scored.append(
            (f, composite, {"relevance": relevance, "recency": recency, "confidence": confidence, "connectivity": conn})
        )
    scored.sort(key=lambda t: t[1], reverse=True)
    return scored


# ──────────────────────────────────────────────────────────────────────────────
# The engine
# ──────────────────────────────────────────────────────────────────────────────


class RetrievalEngine:
    def __init__(
        self,
        searcher: Searcher,
        queries: GraphQueries,
        *,
        redis=None,
        weights: RankWeights | None = None,
        brief_ttl_seconds: int = 1800,
        candidate_multiplier: int = 3,
        min_relevance: float = 0.72,
    ) -> None:
        self.searcher = searcher
        self.queries = queries
        self.redis = redis
        self.weights = weights or RankWeights()
        self.brief_ttl = brief_ttl_seconds
        self.candidate_multiplier = candidate_multiplier
        # Absolute cosine floor. Off-topic queries hover ~0.65–0.70 under BGE-M3;
        # real hits sit ≥ 0.80. 0.72 drops junk without losing a single measured
        # eval hit (calibrated on the live graph, 2026-07). Set 0 to disable.
        self.min_relevance = min_relevance

    async def search(
        self,
        query: str,
        *,
        group_ids: list[str] | None = None,
        limit: int = 10,
        as_of: datetime | None = None,
        center_node_uuid: str | None = None,
    ) -> list[Recalled]:
        """Ranked multi-strategy search over explicit scopes (``None`` = all scopes).

        ``recall`` is this with scopes composed from project/agent; the MCP
        ``search`` tool calls this directly to search across all knowledge.
        """
        # Strategy 1+2: semantic + graph (Graphiti hybrid; center_node_uuid adds BFS).
        multiplier = self.candidate_multiplier
        fetch = limit * multiplier
        raw = await self.searcher.search(query, group_ids, fetch, center_node_uuid)
        # Strategy 3: temporal filter (point-in-time). Archived facts are already
        # dropped by the searcher, so surviving < limit means valid hits may sit
        # past the fetch cap.
        candidates = temporal_filter(raw, as_of)

        # Back-fill (one extra round max): if filtering left us short AND the raw
        # fetch was saturated (more may exist), widen the fetch and refilter.
        if len(candidates) < limit and len(raw) >= fetch:
            fetch = limit * multiplier * 2
            logger.info(
                "recall back-fill: %d/%d valid after filter, re-fetching %d candidates",
                len(candidates), limit, fetch,
            )
            raw = await self.searcher.search(query, group_ids, fetch, center_node_uuid)
            candidates = temporal_filter(raw, as_of)

        # Similarity floor: drop confident-junk candidates whose absolute cosine is
        # below the floor. Off-topic queries return few/no results instead of a full
        # plausible-looking set. Applied after temporal filtering so we never spend
        # the floor budget on already-superseded facts.
        before_floor = len(candidates)
        candidates = apply_similarity_floor(candidates, self.min_relevance)
        if len(candidates) < before_floor:
            logger.info(
                "similarity floor (%.2f) dropped %d/%d low-relevance candidate(s)",
                self.min_relevance, before_floor - len(candidates), before_floor,
            )

        # Connectivity signal from node degrees.
        connectivity = await self._connectivity(candidates)

        ranked = score_facts(candidates, self.weights, connectivity, now=as_of)
        return [
            Recalled(
                fact=f.fact, score=round(s, 4), scope=f.group_id, uuid=f.uuid,
                valid_at=f.valid_at, components={k: round(v, 4) for k, v in c.items()},
            )
            for f, s, c in ranked[:limit]
        ]

    async def recall(
        self,
        query: str,
        *,
        project_id: str | None = None,
        agent_role: str | None = None,
        limit: int = 10,
        as_of: datetime | None = None,
        center_node_uuid: str | None = None,
    ) -> list[Recalled]:
        # global + project (+ agent), composed and ranked together (R5).
        return await self.search(
            query,
            group_ids=Scope.compose(project_id, agent_role),
            limit=limit,
            as_of=as_of,
            center_node_uuid=center_node_uuid,
        )

    async def brief(self, project_id: str, *, use_cache: bool = True) -> Brief:
        key = f"brief:{project_id}"
        if use_cache and self.redis is not None:
            cached = await self.redis.get(key)
            if cached:
                data = json.loads(cached)
                data["cached"] = True
                return Brief(**data)

        scopes = Scope.compose(project_id)  # global + project_<id>

        # Four independent label queries → run concurrently (was serial).
        conventions, decisions, lessons, cross = await asyncio.gather(
            self.queries.nodes_by_label([_CONVENTION], scopes, 7),
            self.queries.nodes_by_label([_DECISION], scopes, 7),
            self.queries.nodes_by_label([_LESSON], scopes, 7),
            self.queries.nodes_by_label([_PATTERN, _DECISION, _CONVENTION], [Scope.GLOBAL], 7),
        )

        brief = Brief(
            project_id=project_id,
            project_summary=self._summarize(project_id, conventions, decisions, lessons),
            active_conventions=[self._line(n) for n in conventions],
            key_decisions=[self._line(n) for n in decisions],
            relevant_lessons=[self._line(n) for n in self._by_severity(lessons)],
            cross_project_knowledge=[self._line(n) for n in cross],
            generated_at=datetime.now(timezone.utc),
        )

        if self.redis is not None:
            await self.redis.set(key, brief.model_dump_json(), ex=self.brief_ttl)
        return brief

    async def invalidate_brief(self, project_id: str) -> None:
        if self.redis is not None:
            await self.redis.delete(f"brief:{project_id}")

    # --- helpers -------------------------------------------------------------

    async def _connectivity(self, facts: list[Fact]) -> dict[str, float]:
        node_uuids = list({u for f in facts for u in (f.source_uuid, f.target_uuid) if u})
        if not node_uuids:
            return {}
        degrees = await self.queries.degrees(node_uuids)
        if not degrees:
            return {}
        max_deg = max(degrees.values()) or 1
        out: dict[str, float] = {}
        for f in facts:
            d = max(degrees.get(f.source_uuid or "", 0), degrees.get(f.target_uuid or "", 0))
            out[f.uuid] = d / max_deg
        return out

    @staticmethod
    def _line(n: NodeRow) -> str:
        return n.summary.strip() if n.summary else n.name

    @staticmethod
    def _by_severity(lessons: list[NodeRow]) -> list[NodeRow]:
        order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        return sorted(lessons, key=lambda n: order.get(str(n.attributes.get("severity", "")).lower(), 4))

    @staticmethod
    def _summarize(project_id: str, conv, dec, les) -> str:
        return (
            f"{project_id}: {len(dec)} key decisions, {len(conv)} active conventions, "
            f"{len(les)} lessons on record."
        )


# ──────────────────────────────────────────────────────────────────────────────
# Real collaborators (wrap Graphiti / Neo4j)
# ──────────────────────────────────────────────────────────────────────────────


class GraphitiSearcher:
    def __init__(self, graphiti) -> None:
        self._graphiti = graphiti

    async def search(self, query, scopes, limit, center_node_uuid=None) -> list[Fact]:
        edges = await self._graphiti.search(
            query, center_node_uuid=center_node_uuid, group_ids=scopes, num_results=limit
        )
        uuids = [e.uuid for e in edges]
        archived = await self._archived_uuids(uuids)
        # Graphiti's high-level search() returns edges ordered by its cross-encoder
        # reranker but attaches NO absolute similarity score (the reranker emits only
        # relative rank positions). An absolute cosine score is what a similarity floor
        # needs — BGE-M3's baseline self-similarity is ~0.65 even for unrelated text,
        # so only a real cosine value separates an on-topic hit (~0.85) from off-topic
        # junk (~0.68). We compute it directly: embed the query once, then batch a
        # single vector.similarity.cosine over the returned edges' fact_embedding.
        cosine = await self._cosine_scores(query, uuids)
        return [
            Fact(
                uuid=e.uuid, fact=e.fact, group_id=e.group_id, created_at=e.created_at,
                valid_at=e.valid_at, invalid_at=e.invalid_at,
                source_uuid=e.source_node_uuid, target_uuid=e.target_node_uuid,
                confidence=(e.attributes or {}).get("confidence") if e.attributes else None,
                score=cosine.get(e.uuid, self._edge_score(e)),
            )
            for e in edges
            if e.uuid not in archived  # curation-archived facts are hidden from retrieval
        ]

    async def _cosine_scores(self, query: str, uuids: list[str]) -> dict[str, float]:
        """Absolute cosine similarity of the query vs each returned edge's fact.

        Returns ``{uuid: cosine}`` for the edges whose ``fact_embedding`` is present.
        Best-effort: any failure (embedder hiccup, missing vector op) yields an empty
        map so the caller falls back to ``_edge_score`` / positional relevance — the
        floor simply doesn't engage rather than dropping everything.
        """
        if not uuids:
            return {}
        try:
            embedded = await self._graphiti.embedder.create(input_data=[query])
        except Exception:  # noqa: BLE001 — best-effort; degrade to no cosine score
            logger.warning("cosine scoring: query embed failed; ranking without a floor", exc_info=True)
            return {}
        # embedder.create may return a flat vector or a list-of-vectors depending on version.
        qv = embedded[0] if embedded and isinstance(embedded[0], (list, tuple)) else embedded
        try:
            res = await self._graphiti.driver.execute_query(
                "MATCH ()-[r:RELATES_TO]->() "
                "WHERE r.uuid IN $uuids AND r.fact_embedding IS NOT NULL "
                "RETURN r.uuid AS uuid, "
                "vector.similarity.cosine(r.fact_embedding, $qv) AS sim",
                uuids=uuids, qv=list(qv),
            )
        except Exception:  # noqa: BLE001 — vector op unavailable / driver error
            logger.warning("cosine scoring: similarity query failed; ranking without a floor", exc_info=True)
            return {}
        return {r["uuid"]: float(r["sim"]) for r in res.records if r["sim"] is not None}

    @staticmethod
    def _edge_score(edge) -> float | None:
        """Best-effort similarity score for a Graphiti search edge.

        Graphiti's reranker attaches an ephemeral score on some versions/configs.
        We read it defensively: a top-level ``score`` attribute first, then an
        ``attributes['score']`` fallback. ``None`` when unavailable → the ranker
        falls back to positional relevance for that result set.
        """
        score = getattr(edge, "score", None)
        if score is None and getattr(edge, "attributes", None):
            score = edge.attributes.get("score")
        if score is None:
            return None
        try:
            return float(score)
        except (TypeError, ValueError):
            return None

    async def _archived_uuids(self, uuids: list[str]) -> set[str]:
        if not uuids:
            return set()
        res = await self._graphiti.driver.execute_query(
            "MATCH ()-[r:RELATES_TO]->() WHERE r.uuid IN $uuids AND r.archived = true "
            "RETURN r.uuid AS uuid",
            uuids=uuids,
        )
        return {r["uuid"] for r in res.records}


class Neo4jGraphQueries:
    def __init__(self, graphiti) -> None:
        self._driver = graphiti.driver

    async def nodes_by_label(self, labels, scopes, limit) -> list[NodeRow]:
        # Only valid, non-archived nodes. Return explicit scalar props — never
        # properties(n), which drags the 1024-dim name_embedding over the wire
        # just to pop it. `severity` is the sole attribute callers consume
        # (brief lesson ordering), so it is surfaced explicitly.
        result = await self._driver.execute_query(
            """
            MATCH (n:Entity)
            WHERE any(l IN labels(n) WHERE l IN $labels)
                  AND n.group_id IN $scopes
                  AND n.invalid_at IS NULL
                  AND coalesce(n.archived, false) = false
            RETURN n.uuid AS uuid, n.name AS name, n.summary AS summary,
                   labels(n) AS labels, n.group_id AS group_id,
                   n.created_at AS created_at, n.severity AS severity
            ORDER BY n.created_at DESC LIMIT $limit
            """,
            labels=labels, scopes=scopes, limit=limit,
        )
        rows = []
        for r in result.records:
            created = r["created_at"]
            if hasattr(created, "to_native"):  # neo4j.time.DateTime -> datetime
                created = created.to_native()
            attributes = {"severity": r["severity"]} if r["severity"] is not None else {}
            rows.append(
                NodeRow(
                    uuid=r["uuid"], name=r["name"], summary=r["summary"],
                    labels=[l for l in r["labels"] if l != "Entity"],
                    group_id=r["group_id"], created_at=created, attributes=attributes,
                )
            )
        return rows

    async def degrees(self, node_uuids) -> dict[str, int]:
        result = await self._driver.execute_query(
            """
            MATCH (n:Entity) WHERE n.uuid IN $uuids
            OPTIONAL MATCH (n)-[r:RELATES_TO]-()
            RETURN n.uuid AS uuid, count(r) AS degree
            """,
            uuids=node_uuids,
        )
        return {r["uuid"]: int(r["degree"]) for r in result.records}


def build_retrieval_engine(graphiti, *, redis=None) -> RetrievalEngine:
    """Wire the real retrieval engine. Pass a redis.asyncio client for brief caching."""
    if redis is None and settings.redis_url:
        import redis.asyncio as aioredis

        redis = aioredis.from_url(settings.redis_url, decode_responses=True)
    return RetrievalEngine(
        searcher=GraphitiSearcher(graphiti),
        queries=Neo4jGraphQueries(graphiti),
        redis=redis,
    )
