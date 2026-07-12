"""Synapse write pipeline (plan Part 4).

CAPTURE -> TRIAGE -> SCOPE -> DEDUPLICATE -> SCORE -> STORE.

Design principle: **don't reimplement Graphiti.** Graphiti's ``add_episode``
already does entity extraction (Claude Sonnet 4.6), embedding (BGE-M3), and edge
creation. This pipeline adds what Graphiti does *not* do:

- **Write-trigger filter (R2):** a cheap Haiku triage decides whether content is
  worth storing at all (decision/convention/lesson/research/pattern/tool) vs
  noise (transcripts, scratch, intermediate reasoning). This is the quality gate.
- **Scope resolution:** map to a Graphiti ``group_id`` (global / project_X /
  agent_Y), defaulting to project, detecting global signals.
- **Pre-store dedup + contradiction:** before storing, compare against existing
  knowledge by vector similarity. ``>= dedup_threshold`` -> duplicate (skip).
  Gray zone -> Haiku adjudicates duplicate / contradiction / distinct.

The LLM (triage), embedder, vector index, and graph are all injected so the
logic is unit-testable without live services (CLAUDE.md §4 conventions).
"""

from __future__ import annotations

import errno
import hashlib
import json
import logging
import re
from datetime import datetime, timezone
from enum import Enum
from typing import Protocol

from pydantic import BaseModel, Field

from graphiti_core.nodes import EpisodeType

from synapse.config import settings
from synapse.core.schema import EDGE_TYPE_MAP, EDGE_TYPES, ENTITY_TYPES, Scope
from synapse.core.vector_index import FACT_VECTOR_INDEX

logger = logging.getLogger("synapse.write_pipeline")

# Content below this length can plausibly produce zero fact edges legitimately (a bare entity mention),
# so an empty extraction there is NOT flagged as degraded. Above it, zero edges is the ~14% local-tail
# silent-failure signature we want surfaced (WP-B item 2).
_DEGRADED_MIN_CHARS = 80

# Name of the native relationship vector index over RELATES_TO.fact_embedding (WP-B item 3). graphiti_core
# does NOT create a vector index on fact_embedding (only fulltext + b-tree on RELATES_TO), so we own it.
# Imported from synapse.core.vector_index so the write pipeline and curation engine share the same literal.
_FACT_VECTOR_DIM = 1024  # BGE-M3, LOCKED (CLAUDE.md §1)

# Valid knowledge types (the schema labels, lowercased, + generic fallback).
KNOWLEDGE_TYPES = ["decision", "convention", "lesson", "research", "pattern", "tool", "entity"]


# ──────────────────────────────────────────────────────────────────────────────
# Result + verdict models
# ──────────────────────────────────────────────────────────────────────────────


class Outcome(str, Enum):
    STORED = "stored"
    DUPLICATE = "duplicate"
    CONTRADICTION = "contradiction"
    REJECTED = "rejected"  # failed the write-trigger filter (R2)


class TriageVerdict(BaseModel):
    """Cheap Haiku classification of incoming content."""

    worth_storing: bool
    knowledge_type: str = "entity"
    is_global: bool = False
    confidence: float = 0.5
    reason: str = ""
    # True when the triage JSON couldn't be parsed. The verdict then FAILS CLOSED
    # (worth_storing=False) so a weak local model's malformed output can't switch the noise
    # filter OFF exactly when it's least trustworthy (WP-B item 1, R2).
    parse_failed: bool = False


class Relation(str, Enum):
    DUPLICATE = "duplicate"
    CONTRADICTION = "contradiction"
    DISTINCT = "distinct"


class Adjudication(BaseModel):
    relation: Relation
    reason: str = ""


class NearestFact(BaseModel):
    uuid: str
    fact: str
    score: float


class WriteResult(BaseModel):
    outcome: Outcome
    reason: str = ""
    knowledge_type: str | None = None
    scope: str | None = None
    confidence: float | None = None
    source: str | None = None
    reference_time: datetime | None = None
    episode_uuid: str | None = None
    entities: list[str] = Field(default_factory=list)
    facts: list[str] = Field(default_factory=list)
    duplicate_of: str | None = None
    contradicts: str | None = None
    # Empty-extraction detection (WP-B item 2): a store that produced zero fact edges for
    # non-trivial content is the local/degraded silent-failure signature. STORED still, but flagged
    # so the caller/UI can surface it. facts_extracted is populated accurately on every path.
    degraded: bool = False
    facts_extracted: int = 0


# ──────────────────────────────────────────────────────────────────────────────
# Injected collaborators (Protocols → mockable)
# ──────────────────────────────────────────────────────────────────────────────


class Embedder(Protocol):
    async def create(self, input_data: str) -> list[float]: ...


class VectorIndex(Protocol):
    async def nearest(self, vec: list[float], scopes: list[str]) -> NearestFact | None: ...


class Triage(Protocol):
    async def classify(self, content: str, hint_type: str | None) -> TriageVerdict: ...
    async def adjudicate(self, new_content: str, existing_fact: str) -> Adjudication: ...


# ──────────────────────────────────────────────────────────────────────────────
# Real implementations
# ──────────────────────────────────────────────────────────────────────────────


class Neo4jVectorIndex:
    """Nearest existing fact by cosine similarity over Graphiti's fact embeddings.

    The dedup lookup runs on EVERY write and was the highest-frequency full scan in the system: a Cypher
    ``vector.similarity.cosine`` over every RELATES_TO edge in scope. This uses Neo4j's NATIVE relationship
    vector index (``db.index.vector.queryRelationships``) instead — an approximate-NN lookup that avoids
    the full scan (WP-B item 3). graphiti_core creates only fulltext/b-tree indexes on RELATES_TO, so we
    create the vector index ourselves (idempotently, once) via :meth:`ensure_index`.

    The old brute-force scan is kept as a fallback: if the index query raises (index still populating,
    unsupported Neo4j build, etc.) we fall back and log ONCE per process, not per call.
    """

    def __init__(self, graphiti) -> None:
        self._driver = graphiti.driver
        self._index_ready = False
        self._fallback_logged = False

    async def ensure_index(self) -> None:
        """Create the relationship vector index idempotently. Safe to call more than once; best-effort."""
        if self._index_ready:
            return
        try:
            await self._driver.execute_query(
                f"""
                CREATE VECTOR INDEX {FACT_VECTOR_INDEX} IF NOT EXISTS
                FOR ()-[e:RELATES_TO]-() ON (e.fact_embedding)
                OPTIONS {{indexConfig: {{
                    `vector.dimensions`: {_FACT_VECTOR_DIM},
                    `vector.similarity_function`: 'cosine'
                }}}}
                """,
            )
            self._index_ready = True
        except Exception as exc:  # noqa: BLE001 — index creation is best-effort; scan fallback still works
            logger.warning(
                "could not create fact vector index %s (%s); dedup uses the brute-force scan",
                FACT_VECTOR_INDEX, str(exc)[:120],
            )

    async def nearest(self, vec: list[float], scopes: list[str]) -> NearestFact | None:
        try:
            return await self._nearest_indexed(vec, scopes)
        except Exception as exc:  # noqa: BLE001 — any index-path failure → fall back to the scan
            if not self._fallback_logged:
                logger.warning(
                    "vector index query failed (%s); falling back to the brute-force cosine scan "
                    "for this process", str(exc)[:120],
                )
                self._fallback_logged = True
            return await self._nearest_scan(vec, scopes)

    async def _nearest_indexed(self, vec: list[float], scopes: list[str]) -> NearestFact | None:
        # Over-fetch a few candidates then scope-filter (the native ANN call can't push down the
        # group_id predicate), returning the top in-scope match. k is small — dedup only needs the best.
        result = await self._driver.execute_query(
            f"""
            CALL db.index.vector.queryRelationships('{FACT_VECTOR_INDEX}', $k, $vec)
            YIELD relationship AS e, score
            WHERE e.group_id IN $scopes AND e.fact_embedding IS NOT NULL
            RETURN e.uuid AS uuid, e.fact AS fact, score
            ORDER BY score DESC LIMIT 1
            """,
            k=25,
            scopes=scopes,
            vec=vec,
        )
        records = result.records
        if not records:
            return None
        r = records[0]
        return NearestFact(uuid=r["uuid"], fact=r["fact"], score=float(r["score"]))

    async def _nearest_scan(self, vec: list[float], scopes: list[str]) -> NearestFact | None:
        result = await self._driver.execute_query(
            """
            MATCH (n:Entity)-[e:RELATES_TO]->(m:Entity)
            WHERE e.group_id IN $scopes AND e.fact_embedding IS NOT NULL
            WITH e, vector.similarity.cosine(e.fact_embedding, $vec) AS score
            RETURN e.uuid AS uuid, e.fact AS fact, score
            ORDER BY score DESC LIMIT 1
            """,
            scopes=scopes,
            vec=vec,
        )
        records = result.records
        if not records:
            return None
        r = records[0]
        return NearestFact(uuid=r["uuid"], fact=r["fact"], score=float(r["score"]))


class ClaudeTriage:
    """Haiku-backed triage + adjudication. Cheap, simple classification only."""

    def __init__(self, api_key: str, model: str) -> None:
        from anthropic import AsyncAnthropic

        self._client = AsyncAnthropic(api_key=api_key)
        self._model = model

    async def _json_call(self, system: str, user: str) -> tuple[dict, bool]:
        # Credit-aware: Haiku when the key has credit, else local gemma (so triage never hard-fails
        # on an exhausted key). Tolerant of accidental prose/fences around the JSON object.
        # Returns (data, parse_failed). parse_failed=True means the JSON could not be recovered — the
        # caller MUST fail closed (WP-B item 1), never treat an unparsable response as permissive.
        from synapse.core.llm_fallback import haiku_or_local

        try:
            text, provider = await haiku_or_local(system, user, max_tokens=400)
        except Exception as exc:  # noqa: BLE001 — an LLM/transport failure is a parse failure downstream
            logger.warning("triage LLM call failed (%s); failing closed", str(exc)[:120])
            return {}, True

        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end == -1 or end < start:
            logger.warning("triage returned no JSON object (provider=%s); failing closed", provider)
            return {}, True
        try:
            return json.loads(text[start : end + 1]), False
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning(
                "triage JSON parse failed (provider=%s: %s); failing closed", provider, str(exc)[:120]
            )
            return {}, True

    async def classify(self, content: str, hint_type: str | None) -> TriageVerdict:
        system = (
            "You are the write-trigger filter for Synapse, a long-term knowledge brain. "
            "STORE only durable knowledge: decisions (with rationale), conventions, lessons "
            "(gotchas/failures/best practices), research findings, reusable patterns, tool "
            "choices. REJECT noise: raw conversation, greetings, intermediate reasoning, "
            "debug scratch, transient state, restated questions. "
            f"knowledge_type must be one of {KNOWLEDGE_TYPES}. is_global is true only when the "
            "knowledge applies across all projects (a personal preference or universal pattern), "
            "false when it is specific to one project. Respond with ONLY a JSON object: "
            '{"worth_storing": bool, "knowledge_type": str, "is_global": bool, '
            '"confidence": float 0-1, "reason": str}.'
        )
        user = content if not hint_type else f"[caller suggests type={hint_type}]\n{content}"
        data, parse_failed = await self._json_call(system, user)
        if parse_failed:
            # FAIL CLOSED (R2): an unparsable verdict must NOT turn the noise filter off. Treat as
            # not-worth-storing; the caller can still `force=True` for a deliberate write.
            return TriageVerdict(
                worth_storing=False,
                knowledge_type=(hint_type or "entity").lower() if (hint_type or "entity").lower() in KNOWLEDGE_TYPES else "entity",
                confidence=0.0,
                reason="triage response could not be parsed (failed closed)",
                parse_failed=True,
            )
        kt = str(data.get("knowledge_type", hint_type or "entity")).lower()
        if kt not in KNOWLEDGE_TYPES:
            kt = "entity"
        return TriageVerdict(
            worth_storing=bool(data.get("worth_storing", True)),
            knowledge_type=kt,
            is_global=bool(data.get("is_global", False)),
            confidence=float(data.get("confidence", 0.5)),
            reason=str(data.get("reason", "")),
        )

    async def adjudicate(self, new_content: str, existing_fact: str) -> Adjudication:
        system = (
            "Compare a NEW piece of knowledge to an EXISTING fact already stored. "
            "Classify their relationship as exactly one of: duplicate (same information), "
            "contradiction (they conflict — one says X, the other not-X), or distinct "
            "(different or merely complementary). Respond with ONLY JSON: "
            '{"relation": "duplicate|contradiction|distinct", "reason": str}.'
        )
        user = f"EXISTING: {existing_fact}\nNEW: {new_content}"
        data, parse_failed = await self._json_call(system, user)
        if parse_failed:
            # Unparsable → treat as DISTINCT: store the new knowledge rather than silently drop it as a
            # duplicate or mis-flag a contradiction on an unreliable verdict (safe under R2).
            return Adjudication(relation=Relation.DISTINCT, reason="adjudication unparsable (failed to distinct)")
        try:
            relation = Relation(str(data.get("relation", "distinct")).lower())
        except ValueError:
            relation = Relation.DISTINCT
        return Adjudication(relation=relation, reason=str(data.get("reason", "")))


# ──────────────────────────────────────────────────────────────────────────────
# The pipeline
# ──────────────────────────────────────────────────────────────────────────────


class WritePipeline:
    def __init__(
        self,
        graphiti,
        embedder: Embedder,
        index: VectorIndex,
        triage: Triage,
        *,
        dedup_threshold: float | None = None,
        relate_floor: float | None = None,
    ) -> None:
        self.graphiti = graphiti
        self.embedder = embedder
        self.index = index
        self.triage = triage
        self.dedup_threshold = dedup_threshold if dedup_threshold is not None else settings.dedup_threshold
        self.relate_floor = relate_floor if relate_floor is not None else settings.relate_floor
        self._index_ensured = False
        self._health_checked = False

    async def _ensure_index(self) -> None:
        """Idempotently create the native fact vector index once, on the first write (WP-B item 3).

        Done lazily here (not in the sync builder) because index DDL is async. No-op when the index
        implementation doesn't support it (unit-test fakes).
        """
        if self._index_ensured:
            return
        ensure = getattr(self.index, "ensure_index", None)
        if ensure is not None:
            await ensure()
        self._index_ensured = True

    async def _check_embedder_health(self) -> None:
        """Cheap startup probe: GET the Ollama base URL; logs a WARNING if unreachable.

        Never raises. Never blocks the pipeline. Called lazily on the first write (flag guard).
        The ``settings.ollama_base_url`` ends in ``/v1`` for Ollama's OpenAI-compatible endpoint;
        we strip the path suffix and probe the root to avoid a false 404 on the versioned path.
        """
        if self._health_checked:
            return
        self._health_checked = True
        try:
            import httpx  # soft dependency; already a transitive dep via graphiti/anthropic

            base = settings.ollama_base_url.rstrip("/")
            # Strip the /v1 OpenAI-compat suffix so we probe the Ollama root (returns 200 always).
            probe_url = base.removesuffix("/v1") if base.endswith("/v1") else base
            async with httpx.AsyncClient(timeout=2.0) as client:
                resp = await client.get(probe_url)
                resp.raise_for_status()
        except Exception as exc:  # noqa: BLE001 — health probe must never raise
            logger.warning(
                "Ollama embedder unreachable at %s — writes will be queued for replay (%s)",
                settings.ollama_base_url, str(exc)[:120],
            )

    async def remember(
        self,
        content: str,
        *,
        knowledge_type: str | None = None,
        project_id: str | None = None,
        agent_role: str | None = None,
        source: str = "agent",
        reference_time: datetime | None = None,
        force: bool = False,
    ) -> WriteResult:
        content = content.strip()
        reference_time = reference_time or datetime.now(timezone.utc)

        # Startup health probe (once, non-blocking, no raise).
        await self._check_embedder_health()

        # 1+2. CAPTURE + TRIAGE (write-trigger filter, R2)
        verdict = await self.triage.classify(content, knowledge_type)
        if not verdict.worth_storing and not force:
            return WriteResult(
                outcome=Outcome.REJECTED,
                reason=verdict.reason or "did not pass the write-trigger filter",
                knowledge_type=verdict.knowledge_type,
            )
        ktype = (knowledge_type or verdict.knowledge_type).lower()

        # 3. SCOPE — explicit project wins; else global signal; else require a project.
        scope = self._resolve_scope(verdict, project_id, agent_role)

        # 4a. CONTENT-HASH GUARD (WP-B item 4) — a deterministic, race-free exact-duplicate check that
        # runs BEFORE the probabilistic vector compare. sha256 over whitespace-normalized content; an
        # exact match on an already-stored Episode in the same group_id is an unambiguous duplicate
        # (the vector path compares content-embeddings to fact-embeddings — different provenance — and
        # is check-then-act racy under concurrent writes, so it can't catch identical re-submits reliably).
        chash = _content_hash(content)
        existing_hash_uuid = await self._existing_hash(chash, scope)
        if existing_hash_uuid is not None and not force:
            return WriteResult(
                outcome=Outcome.DUPLICATE,
                reason=f"exact content-hash match (sha256) vs episode {existing_hash_uuid}",
                knowledge_type=ktype,
                scope=scope,
                duplicate_of=existing_hash_uuid,
            )

        # 4b. DEDUPLICATE — compare against existing knowledge in this scope (+ global).
        await self._ensure_index()
        try:
            vec = await self.embedder.create(content)
        except Exception as exc:  # noqa: BLE001
            if _is_embedder_down(exc):
                logger.warning(
                    "embedder unavailable during dedup probe (%s); queueing write for replay",
                    str(exc)[:120],
                )
                await self._queue_for_replay(content, scope, ktype, reference_time)
                return WriteResult(
                    outcome=Outcome.STORED,
                    reason="queued: embedder unavailable",
                    knowledge_type=ktype,
                    scope=scope,
                    degraded=True,
                    facts_extracted=0,
                )
            raise

        nearest = await self.index.nearest(vec, self._dedup_scopes(scope))
        duplicate_of: str | None = None
        contradicts: str | None = None

        if nearest is not None and nearest.score >= self.dedup_threshold:
            return WriteResult(
                outcome=Outcome.DUPLICATE,
                reason=f"cosine {nearest.score:.3f} >= {self.dedup_threshold} vs {nearest.uuid}",
                knowledge_type=ktype,
                scope=scope,
                duplicate_of=nearest.uuid,
            )

        if nearest is not None and nearest.score >= self.relate_floor:
            verdict_rel = await self.triage.adjudicate(content, nearest.fact)
            if verdict_rel.relation is Relation.DUPLICATE:
                return WriteResult(
                    outcome=Outcome.DUPLICATE,
                    reason=verdict_rel.reason or "adjudged duplicate",
                    knowledge_type=ktype,
                    scope=scope,
                    duplicate_of=nearest.uuid,
                )
            if verdict_rel.relation is Relation.CONTRADICTION:
                # Flag it, but still STORE the new truth — Graphiti's temporal model
                # supersedes the old fact. Persisting an explicit Contradicts edge +
                # surfacing in the Curate UI is later (curation engine / Phase 9).
                contradicts = nearest.uuid

        # 5+6. SCORE + STORE (Graphiti does extraction + embedding + edges)
        try:
            result = await self._store(content, scope, source, reference_time)
        except Exception as exc:  # noqa: BLE001
            if _is_embedder_down(exc):
                logger.warning(
                    "embedder unavailable during store (%s); queueing write for replay",
                    str(exc)[:120],
                )
                await self._queue_for_replay(content, scope, ktype, reference_time)
                return WriteResult(
                    outcome=Outcome.STORED,
                    reason="queued: embedder unavailable",
                    knowledge_type=ktype,
                    scope=scope,
                    degraded=True,
                    facts_extracted=0,
                )
            raise
        episode_uuid = getattr(getattr(result, "episode", None), "uuid", None)

        # Stamp the content hash on the stored Episode so the 4a guard catches future re-submits.
        await self._stamp_hash(episode_uuid, chash)

        facts = [e.fact for e in getattr(result, "edges", []) or []]
        facts_extracted = len(facts)

        # WP-B item 2 — EMPTY-EXTRACTION DETECTION. Zero fact edges for non-trivial content is the
        # local/degraded silent-failure signature (~14% of dense writes). Still STORED, but flagged +
        # logged (with which extraction provider ran) and, if possible, queued for review.
        degraded = facts_extracted == 0 and len(content) > _DEGRADED_MIN_CHARS
        if degraded:
            provider = settings.extraction_mode
            logger.warning(
                "empty extraction: 0 fact edges for %d-char content (extraction_mode=%s, scope=%s, "
                "episode=%s) — stored but degraded, queued for review",
                len(content), provider, scope, episode_uuid,
            )
            await self._queue_degraded(content, scope, ktype, episode_uuid)

        return WriteResult(
            outcome=Outcome.CONTRADICTION if contradicts else Outcome.STORED,
            reason="stored (supersedes a contradicting fact)" if contradicts else "stored",
            knowledge_type=ktype,
            scope=scope,
            confidence=verdict.confidence,
            source=source,
            reference_time=reference_time,
            episode_uuid=episode_uuid,
            entities=[n.name for n in getattr(result, "nodes", []) or []],
            facts=facts,
            duplicate_of=duplicate_of,
            contradicts=contradicts,
            degraded=degraded,
            facts_extracted=facts_extracted,
        )

    # --- helpers -------------------------------------------------------------

    def _resolve_scope(self, verdict: TriageVerdict, project_id: str | None, agent_role: str | None) -> str:
        # Explicit scope ALWAYS wins (R5). An explicit project_id is a hard signal — the
        # triage's is_global guess must NOT override it (it over-flags project-specific
        # knowledge as global, leaking it cross-project and polluting every brief, R7).
        # Promotion to global is a deliberate curation action, not a per-write LLM guess
        # (the Curate panel surfaces cross-project recurrences as promotion candidates).
        if agent_role:
            return Scope.agent(agent_role)
        if project_id is not None:
            return Scope.project(project_id)
        # No project/agent context: global is the only place it can go (is_global moot here).
        return Scope.GLOBAL

    def _dedup_scopes(self, scope: str) -> list[str]:
        # Dedup ONLY within the target scope (R7). Deduping a project write against
        # global discarded project-contextualized restatements ("Acme-API uses BigDecimal
        # for monetary/price values") that recall from the project's seat needs — the
        # global version ("BigDecimal for money in Java") ranks low for an Acme-API-phrased
        # query. Cross-scope redundancy is the curation engine's job (fact-to-fact dedup +
        # promotion candidates), not the write path's.
        return [scope]

    async def _store(self, content: str, scope: str, source: str, reference_time: datetime):
        return await self.graphiti.add_episode(
            name=f"synapse-{int(reference_time.timestamp())}",
            episode_body=content,
            source=EpisodeType.text,
            source_description=f"synapse:{source}",
            reference_time=reference_time,
            group_id=scope,
            entity_types=ENTITY_TYPES,
            edge_types=EDGE_TYPES,
            edge_type_map=EDGE_TYPE_MAP,
        )

    # --- content-hash dedup (WP-B item 4) -----------------------------------

    def _driver(self):
        """The Neo4j driver, or None when the graph backend doesn't expose one (unit tests)."""
        return getattr(self.graphiti, "driver", None)

    async def _existing_hash(self, content_hash: str, scope: str) -> str | None:
        """UUID of an already-stored Episodic node with this exact content hash in the same group_id.

        Best-effort: any driver error (or no driver, as in unit tests) means "no exact match found" and
        the write falls through to the normal vector dedup — the hash guard never blocks a legitimate write.
        """
        driver = self._driver()
        if driver is None:
            return None
        try:
            result = await driver.execute_query(
                """
                MATCH (e:Episodic {content_hash: $h, group_id: $scope})
                RETURN e.uuid AS uuid LIMIT 1
                """,
                h=content_hash, scope=scope,
            )
            records = result.records
            return records[0]["uuid"] if records else None
        except Exception as exc:  # noqa: BLE001 — hash guard is an optimization; never fail the write
            logger.debug("content-hash lookup failed (%s); skipping the exact-dup guard", str(exc)[:120])
            return None

    async def _stamp_hash(self, episode_uuid: str | None, content_hash: str) -> None:
        """Persist content_hash on the stored Episodic node so future re-submits hit the 4a guard."""
        driver = self._driver()
        if driver is None or episode_uuid is None:
            return
        try:
            await driver.execute_query(
                "MATCH (e:Episodic {uuid: $uuid}) SET e.content_hash = $h",
                uuid=episode_uuid, h=content_hash,
            )
        except Exception as exc:  # noqa: BLE001 — a missed stamp only weakens future exact-dup detection
            logger.debug("could not stamp content_hash on episode %s (%s)", episode_uuid, str(exc)[:120])

    async def _queue_for_replay(
        self,
        content: str,
        scope: str,
        ktype: str,
        reference_time: datetime,
    ) -> None:
        """Dead-letter a write that failed because the embedder/Ollama is down.

        Creates (or merges) a PendingCapture node with status='pending_replay'. The replay
        Celery task scans for these and retries the full remember() pipeline when Ollama
        comes back up.

        IMPORTANT: this path must NOT call the embedder — it stores only to Neo4j (the
        driver's execute_query). If Neo4j is also down, the execute_query raises and we
        let it hard-fail (do not swallow it here).
        """
        driver = self._driver()
        if driver is None:
            logger.warning("no Neo4j driver — cannot queue pending_replay for scope=%s", scope)
            return
        h = _content_hash(content)
        await driver.execute_query(
            """
            MERGE (p:PendingCapture {hash: $h})
            ON CREATE SET p.uuid = randomUUID(),
                p.project_id = $scope,
                p.content = $content,
                p.type = $type,
                p.status = 'pending_replay',
                p.retry_count = 0,
                p.reference_time = $reference_time,
                p.created_at = datetime()
            ON MATCH SET p.status = 'pending_replay'
            """,
            h=h,
            scope=scope,
            content=content,
            type=ktype,
            reference_time=reference_time.isoformat(),
        )
        logger.info(
            "queued pending_replay: scope=%s type=%s hash=%.12s", scope, ktype, h
        )

    async def _queue_degraded(self, content: str, scope: str, ktype: str, episode_uuid: str | None) -> None:
        """Queue a degraded (empty-extraction) write to the review queue (WP-B item 2).

        The pending-capture queue lives in session_capture and imports only config at module level, so
        this reuse creates no import cycle. Best-effort: any failure leaves the flag+log as the signal.
        """
        driver = self._driver()
        if driver is None:
            return
        try:
            h = _content_hash(content)
            await driver.execute_query(
                """
                MERGE (p:PendingCapture {hash: $h})
                ON CREATE SET p.uuid = randomUUID(), p.project_id = $scope, p.content = $content,
                    p.type = $type, p.confidence = 0.0, p.reason = 'degraded write: 0 fact edges extracted',
                    p.episode_uuid = $episode_uuid, p.created_at = datetime(), p.status = 'degraded'
                """,
                h=h, scope=scope, content=content, type=ktype, episode_uuid=episode_uuid,
            )
        except Exception as exc:  # noqa: BLE001 — queueing is best-effort; the flag+log already fired
            logger.debug("could not queue degraded write for review (%s)", str(exc)[:120])


def _is_embedder_down(exc: BaseException) -> bool:
    """Return True when *exc* indicates the Ollama embedder is unreachable (not a logic error).

    Detects:
    - httpx.ConnectError / httpx.ConnectTimeout (the OpenAI-compat client uses httpx)
    - aiohttp.ClientConnectionError (Graphiti's internal HTTP layer)
    - stdlib ConnectionRefusedError / OSError(ECONNREFUSED)
    - Any exception whose string contains "connection refused" or "ollama" (case-insensitive)

    Everything else (extraction errors, schema errors, etc.) is NOT an embedder-down
    signal and must hard-fail so logic bugs are surfaced, not silently queued.
    """
    # stdlib connection errors
    if isinstance(exc, ConnectionRefusedError):
        return True
    if isinstance(exc, OSError) and getattr(exc, "errno", None) == errno.ECONNREFUSED:
        return True

    # httpx (soft-dep, already pulled in transitively)
    try:
        import httpx
        if isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout)):
            return True
    except ImportError:
        pass

    # aiohttp (used internally by some Graphiti versions)
    try:
        import aiohttp
        if isinstance(exc, aiohttp.ClientConnectionError):
            return True
    except ImportError:
        pass

    # String-based fallback: covers nested/wrapped exceptions
    msg = str(exc).lower()
    if "connection refused" in msg or "ollama" in msg:
        return True

    return False


def _content_hash(content: str) -> str:
    """sha256 of whitespace-normalized content — the deterministic exact-duplicate key (WP-B item 4).

    Collapses all runs of whitespace to single spaces and strips, so cosmetically different re-submits
    of the same knowledge (added newline, trailing space, reindent) hash identically.
    """
    normalized = re.sub(r"\s+", " ", content).strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def build_write_pipeline(graphiti) -> WritePipeline:
    """Wire the real pipeline onto a constructed Graphiti instance."""
    from synapse.core.knowledge_engine import build_embedder

    return WritePipeline(
        graphiti=graphiti,
        embedder=build_embedder(),
        index=Neo4jVectorIndex(graphiti),
        triage=ClaudeTriage(settings.anthropic_api_key, settings.triage_model),
    )
