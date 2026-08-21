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
# Pure functions only — the deterministic lexical rules live beside their siblings and their measured
# rationale in consolidation_engine; importing them does not pull the consolidation engine itself
# into the write path (that module imports only synapse.core.schema).
from synapse.core.consolidation_engine import (
    could_replace,
    invalidation_is_credible,
    subject_overlap,
)
from synapse.core.injection import looks_like_instruction
from synapse.core.provenance import Provenance
from synapse.core.redaction import redact
from synapse.core.schema import (
    EDGE_TYPE_MAP,
    EDGE_TYPES,
    ENTITY_TYPES,
    Scope,
    canonical_edge_name,
)
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

# --- global-write gate (research §5.3, roadmap item 15) ---------------------------------------
# `global` is the one scope retrieval composes for EVERY project, and since the UserPromptSubmit
# hook it is injected into every prompt in every project — so a project-specific fact stored there
# is noise eleven times over. Measured on the live graph 2026-07-25: of 137 active global facts,
# **41 (30%) name exactly one project**, e.g. "The decision to use ib_async instead of ib_insync
# applies to the Acme-Sim project". Worse, that knowledge often appears TWICE in global, once per
# project ("...applies to Acme-API"), when what it really is is trading-domain knowledge that belongs
# in cluster_trading.
#
# The gate REDIRECTS rather than rejects. Rejecting would be safer for global but loses the
# knowledge whenever an agent does not retry; redirecting keeps it and files it correctly, and the
# result reports the redirect so the caller learns.
_TRUSTED_GLOBAL_SOURCES = frozenset({
    # The consolidation engine only reaches global through a reviewed promotion, whose whole purpose
    # is to place evidence-backed cross-project knowledge there. Gating it would fight itself.
    "consolidation",
})


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
    # Credential kinds stripped from the content before it was embedded/stored (research §5.1).
    # Kinds only — never the values. Empty on the overwhelmingly common clean path.
    redactions: list[str] = Field(default_factory=list)
    # Set when the global-write gate refiled this write (research §5.3). Empty on every other path.
    scope_redirected_from: str | None = None
    # Instruction-shaped content detected on the way into a broadcast scope. Names the patterns,
    # never quotes the payload — same rule as `redactions` above.
    injection_kinds: list[str] = Field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────────────
# Injected collaborators (Protocols → mockable)
# ──────────────────────────────────────────────────────────────────────────────


class Embedder(Protocol):
    async def create(self, input_data: str) -> list[float]: ...


class VectorIndex(Protocol):
    async def nearest(self, vec: list[float], scopes: list[str]) -> NearestFact | None: ...
    async def nearest_k(
        self, vec: list[float], scopes: list[str], k: int
    ) -> list[NearestFact]: ...


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

    async def nearest_k(self, vec: list[float], scopes: list[str], k: int) -> list[NearestFact]:
        """The k most similar in-scope facts, descending. Used for top-k adjudication (item 16)."""
        try:
            return await self._nearest_k_indexed(vec, scopes, k)
        except Exception as exc:  # noqa: BLE001 — index path failure -> brute-force scan
            if not self._fallback_logged:
                logger.warning(
                    "vector index top-k query failed (%s); falling back to the cosine scan "
                    "for this process", str(exc)[:120],
                )
                self._fallback_logged = True
            return await self._nearest_k_scan(vec, scopes, k)

    async def _nearest_k_indexed(self, vec, scopes, k) -> list[NearestFact]:
        # Over-fetch before the scope filter, which the native ANN call cannot push down.
        result = await self._driver.execute_query(
            f"""
            CALL db.index.vector.queryRelationships('{FACT_VECTOR_INDEX}', $fetch, $vec)
            YIELD relationship AS e, score
            WHERE e.group_id IN $scopes AND e.fact_embedding IS NOT NULL
              AND e.invalid_at IS NULL AND coalesce(e.archived, false) = false
            RETURN e.uuid AS uuid, e.fact AS fact, score
            ORDER BY score DESC LIMIT $k
            """,
            fetch=max(25, k * 5), scopes=scopes, vec=vec, k=k,
        )
        return [
            NearestFact(uuid=r["uuid"], fact=r["fact"], score=float(r["score"]))
            for r in result.records
        ]

    async def _nearest_k_scan(self, vec, scopes, k) -> list[NearestFact]:
        result = await self._driver.execute_query(
            """
            MATCH (n:Entity)-[e:RELATES_TO]->(m:Entity)
            WHERE e.group_id IN $scopes AND e.fact_embedding IS NOT NULL
              AND e.invalid_at IS NULL AND coalesce(e.archived, false) = false
            WITH e, vector.similarity.cosine(e.fact_embedding, $vec) AS score
            RETURN e.uuid AS uuid, e.fact AS fact, score
            ORDER BY score DESC LIMIT $k
            """,
            scopes=scopes, vec=vec, k=k,
        )
        return [
            NearestFact(uuid=r["uuid"], fact=r["fact"], score=float(r["score"]))
            for r in result.records
        ]

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
            "(different or merely complementary). "
            # The decisive test, spelled out because the local model got this wrong on real data:
            # it flagged "Haiku handles the sentiment analysis node" vs "...the technical analysis
            # node" as contradictory, when both are simply true. Different members of a set are
            # DISTINCT; only a single-valued attribute can actually conflict.
            "DECISIVE TEST for contradiction: could BOTH statements be true at the same time? "
            "If yes, they are distinct, NOT a contradiction. Two facts naming different members "
            'of a set are distinct ("handles the sentiment node" vs "handles the technical node" '
            "are both true). Only call it a contradiction when the statements are mutually "
            'exclusive — the same single-valued thing given two different values ("runs on port '
            '8080" vs "runs on port 9090"). '
            "Respond with ONLY JSON: "
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
        known_projects=None,
        cluster_resolver=None,
        candidate_k: int | None = None,
        max_adjudications: int | None = None,
    ) -> None:
        self.graphiti = graphiti
        self.embedder = embedder
        self.index = index
        self.triage = triage
        self.dedup_threshold = dedup_threshold if dedup_threshold is not None else settings.dedup_threshold
        self.relate_floor = relate_floor if relate_floor is not None else settings.relate_floor
        self._index_ensured = False
        self._health_checked = False
        # Injected (not imported) so the pipeline stays registry-agnostic and unit-testable, matching
        # RetrievalEngine.cluster_resolver. `known_projects` is a callable returning project ids.
        self._known_projects = known_projects
        self._cluster_resolver = cluster_resolver
        # How many neighbours to consider, and how many of those may cost an LLM adjudication.
        # Fetching is one cheap ANN query; judging is not, so they are separate knobs.
        self.candidate_k = (
            candidate_k if candidate_k is not None else settings.adjudication_candidates
        )
        self.max_adjudications = (
            max_adjudications if max_adjudications is not None else settings.max_adjudications
        )

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
        cluster: str | None = None,
        source: str = "agent",
        reference_time: datetime | None = None,
        force: bool = False,
        dedup_scopes: list[str] | None = None,
        provenance: Provenance | None = None,
    ) -> WriteResult:
        """Store knowledge, stripping any credentials first (research §5.1).

        Redaction is a WRAPPER around the pipeline rather than a step inside it so that no
        return path can accidentally bypass it: every write, on every outcome, reports what was
        stripped. It runs before triage (an outbound LLM call), before the embedder, and before
        the content hash — so a credential never reaches an API, the vector index, or the graph.
        """
        content, redactions = redact(content.strip())
        if redactions:
            # Kinds only — logging the value would just move the leak into the log file.
            logger.warning(
                "redacted %s from an incoming write (scope=%s, source=%s); "
                "the surrounding knowledge was kept",
                ", ".join(redactions), project_id or agent_role or "global", source,
            )
        result = await self._pipeline(
            content,
            knowledge_type=knowledge_type,
            project_id=project_id,
            agent_role=agent_role,
            cluster=cluster,
            source=source,
            reference_time=reference_time,
            force=force,
            dedup_scopes=dedup_scopes,
            provenance=provenance,
        )
        result.redactions = redactions
        return result

    async def _pipeline(
        self,
        content: str,
        *,
        knowledge_type: str | None = None,
        project_id: str | None = None,
        agent_role: str | None = None,
        cluster: str | None = None,
        source: str = "agent",
        reference_time: datetime | None = None,
        force: bool = False,
        dedup_scopes: list[str] | None = None,
        provenance: Provenance | None = None,
    ) -> WriteResult:
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
        scope = self._resolve_scope(verdict, project_id, agent_role, cluster)
        # GLOBAL-WRITE GATE (roadmap item 15): global is injected into every project's prompts, so a
        # project-specific fact there is noise everywhere. Redirect it to where it belongs.
        redirected_from: str | None = None
        if scope == Scope.GLOBAL and source not in _TRUSTED_GLOBAL_SOURCES:
            better = self._better_scope_than_global(content)
            if better:
                logger.info("global-write gate: refiling to %s (source=%s)", better, source)
                redirected_from, scope = Scope.GLOBAL, better

        # INSTRUCTION GATE. The broadcast scopes are the blast radius: a global fact is injected
        # into every project's every prompt by the session hook, so instruction-shaped text stored
        # there is read as an order by an agent that has no way to tell it was authored by another
        # model rather than by the operator. Checked here, at the admission chokepoint, rather than
        # at every read.
        # Only the broadcast scopes are gated. A project-scoped fact already reaches exactly one
        # codebase, which is the containment this gate would otherwise be trying to achieve — and
        # `_resolve_scope` gives an explicit project_id priority over the global signal, so
        # reaching a broadcast scope here means no project was supplied to file it under.
        # Refusal is therefore the only move left, and the reason names the way out.
        injection = looks_like_instruction(content)
        broadcast = {Scope.GLOBAL} | ({Scope.cluster(cluster)} if cluster else set())
        if injection and scope in broadcast:
            logger.warning(
                "instruction gate: refused a %s write (%s)", scope, ",".join(injection.kinds),
            )
            return WriteResult(
                outcome=Outcome.REJECTED,
                reason=(
                    "reads as an instruction to an assistant rather than a statement about "
                    f"code ({', '.join(injection.kinds)}), and targets a scope that is injected "
                    "into every project's prompts. Re-submit with a project_id to contain it to "
                    "one codebase, or rephrase it as a statement."
                ),
                knowledge_type=ktype,
                scope=scope,
                injection_kinds=injection.kinds,
            )

        # 4a. CONTENT-HASH GUARD (WP-B item 4) — a deterministic, race-free exact-duplicate check that
        # runs BEFORE the probabilistic vector compare. sha256 over whitespace-normalized content; an
        # exact match on an already-stored Episode in the same group_id is an unambiguous duplicate
        # (the vector path compares content-embeddings to fact-embeddings — different provenance — and
        # is check-then-act racy under concurrent writes, so it can't catch identical re-submits reliably).
        # Scopes both duplicate guards compare against. Defaults to the write's own scope; a
        # caller may widen it deliberately (see _dedup_scopes).
        compare_scopes = self._dedup_scopes(scope, dedup_scopes)
        chash = _content_hash(content)
        existing_hash_uuid = await self._existing_hash(chash, compare_scopes)
        if existing_hash_uuid is not None and not force:
            return WriteResult(
                outcome=Outcome.DUPLICATE,
                reason=f"exact content-hash match (sha256) vs episode {existing_hash_uuid}",
                knowledge_type=ktype,
                scope=scope,
                duplicate_of=existing_hash_uuid,
                scope_redirected_from=redirected_from,
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

        # TOP-K, not just the nearest (roadmap item 16). Adjudicating only the single closest fact
        # meant a write contradicting the SECOND-nearest was never flagged — which is why the live
        # graph holds just 7 Contradicts edges across 3,039. The deterministic dedup below still
        # only needs the closest; the gray band is where judgement is required, and there may be
        # several candidates in it.
        candidates = await self._nearest_candidates(vec, compare_scopes)
        nearest = candidates[0] if candidates else None
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

        # Gray band, descending similarity. A DUPLICATE verdict wins immediately (nothing new to
        # store); contradictions are remembered but the scan continues, because a later candidate
        # may reveal the write is a duplicate after all — and storing a duplicate is worse than
        # missing one contradiction flag. Adjudications are capped: each is an LLM call.
        gray = [c for c in candidates if self.relate_floor <= c.score < self.dedup_threshold]
        verdict_rel = None
        for candidate in gray[: self.max_adjudications]:
            verdict_rel = await self.triage.adjudicate(content, candidate.fact)
            if verdict_rel.relation is Relation.DUPLICATE:
                return WriteResult(
                    outcome=Outcome.DUPLICATE,
                    reason=verdict_rel.reason or "adjudged duplicate",
                    knowledge_type=ktype,
                    scope=scope,
                    duplicate_of=candidate.uuid,
                )
            if verdict_rel.relation is Relation.CONTRADICTION and contradicts is None:
                # Keep the highest-similarity contradiction (the list is descending).
                contradicts = candidate.uuid
        if len(gray) > self.max_adjudications:
            logger.info(
                "adjudication capped at %d of %d gray-band candidates for this write",
                self.max_adjudications, len(gray),
            )

        # A contradiction does NOT block the write: Graphiti's temporal model supersedes the old
        # fact, and refusing the new truth would be exactly backwards (R4). It is flagged, and now
        # also PERSISTED as an explicit Contradicts edge after the store, so it is visible to the
        # graph and the Curate panel rather than living only in this response.

        # 5+6. SCORE + STORE (Graphiti does extraction + embedding + edges)
        #
        # Taken BEFORE the store so the guard below can tell which edges *this* write expired.
        # Graphiti preserves an existing `expired_at` rather than overwriting it, so an edge whose
        # `expired_at` is at or after this instant provably had none beforehand — i.e. it was live.
        store_started_at = datetime.now(timezone.utc)
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

        # Stamp the content hash (so the 4a guard catches future re-submits) and the write's
        # provenance (roadmap item 13) in one round trip.
        await self._stamp_episode(episode_uuid, chash, provenance)

        if contradicts:
            await self._persist_contradiction(episode_uuid, contradicts)

        # Fold freshly-extracted edge names onto the schema vocabulary (research §2.2). PREVENTION:
        # the extractor invents a new relation name per episode, which is how the live graph reached
        # 535 distinct names over 3,018 edges with 328 names used exactly once. Normalizing on write
        # stops that growing; scripts/normalize_edge_names.py cleans up what predates this.
        await self._canonicalize_edge_names(result)

        facts = [e.fact for e in getattr(result, "edges", []) or []]
        facts_extracted = len(facts)

        # Undo any invalidation Graphiti made on a fact this write did not actually contradict.
        # Runs on EVERY write because the damage is a side effect of the normal path, not of curation.
        # `uuid` via getattr: every real Graphiti EntityEdge has one, but the graph backend is a
        # seam the tests fill with minimal doubles, and an audit must never be what breaks a write.
        own_edges = getattr(result, "edges", []) or []
        await self._revert_unjustified_invalidations(
            since=store_started_at,
            new_facts=facts,
            own_edge_uuids=[
                uuid for uuid in (getattr(e, "uuid", None) for e in own_edges) if uuid
            ],
            # (source, target, relation) for each edge this write created — what the structural
            # half of the veto compares against. Same getattr defensiveness as the uuids above:
            # a real Graphiti EntityEdge has all three, but the graph backend is a seam the tests
            # fill with minimal doubles, and an audit must never be what breaks a write. An edge
            # missing any of them is dropped rather than compared against None.
            new_edges=[
                (src, dst, name) for src, dst, name in (
                    (getattr(e, "source_node_uuid", None), getattr(e, "target_node_uuid", None),
                     getattr(e, "name", None))
                    for e in own_edges
                ) if src and dst
            ],
        )

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
            reason=(
                f"stored (global-write gate refiled this to {scope})" if redirected_from
                else "stored (supersedes a contradicting fact)" if contradicts
                else "stored"
            ),
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
            scope_redirected_from=redirected_from,
            # Non-empty when the content read as an instruction. On this path it was contained to
            # a project rather than refused, so the caller should still see why its reach shrank.
            injection_kinds=injection.kinds,
        )

    # --- helpers -------------------------------------------------------------

    def _resolve_scope(self, verdict: TriageVerdict, project_id: str | None,
                       agent_role: str | None, cluster: str | None = None) -> str:
        # Explicit scope ALWAYS wins (R5). An explicit project_id is a hard signal — the
        # triage's is_global guess must NOT override it (it over-flags project-specific
        # knowledge as global, leaking it cross-project and polluting every brief, R7).
        # Promotion to global is a deliberate curation action, not a per-write LLM guess
        # (the Curate panel surfaces cross-project recurrences as promotion candidates).
        if agent_role:
            return Scope.agent(agent_role)
        # An explicit cluster is likewise deliberate: the caller is saying "this belongs to the
        # whole domain, not just this project". It outranks project_id because it is the more
        # specific instruction — a caller passes both when writing domain knowledge from within
        # a project. It does NOT outrank agent_role, which is the narrowest scope of all.
        if cluster:
            return Scope.cluster(cluster)
        if project_id is not None:
            return Scope.project(project_id)
        # No project/agent context: global is the only place it can go (is_global moot here).
        return Scope.GLOBAL

    def _better_scope_than_global(self, content: str) -> str | None:
        """Instance wrapper over :func:`better_scope_than_global` using the injected resolvers."""
        if self._known_projects is None:
            return None
        try:
            projects = list(self._known_projects() or [])
        except Exception:  # noqa: BLE001 — an unreadable registry must not block a write
            logger.warning("global-write gate: project lookup failed; allowing global", exc_info=True)
            return None
        return better_scope_than_global(content, projects, self._cluster_resolver)

    async def _nearest_candidates(self, vec: list[float], scopes: list[str]) -> list[NearestFact]:
        """Top-k similar facts, descending.

        Falls back to a single ``nearest`` when the index implementation predates top-k (unit-test
        fakes, older builds), so adjudication degrades to the old behaviour instead of failing.
        """
        fetch = getattr(self.index, "nearest_k", None)
        if fetch is not None:
            try:
                return list(await fetch(vec, scopes, self.candidate_k) or [])
            except Exception:  # noqa: BLE001 — degrade to top-1 rather than fail the write
                logger.warning("top-k candidate lookup failed; falling back to nearest",
                               exc_info=True)
        one = await self.index.nearest(vec, scopes)
        return [one] if one is not None else []

    async def _persist_contradiction(self, episode_uuid: str | None, contradicted_uuid: str) -> None:
        """Mark the superseded fact as contradicted, pointing at the fact that replaced it.

        Before this, a detected contradiction lived only in the write response — which is why the
        live graph held 7 Contradicts edges against 3,039 facts. Stamping the OLD edge (rather than
        creating a new relationship between entity nodes) keeps this a property write: it cannot
        alter graph structure, and the Curate panel can find it with one predicate.

        Best-effort — the write has already succeeded by this point and must not be undone.
        """
        driver = self._driver()
        if driver is None or episode_uuid is None:
            return
        try:
            await driver.execute_query(
                """
                MATCH ()-[old:RELATES_TO {uuid: $old}]->()
                OPTIONAL MATCH ()-[new:RELATES_TO]->()
                    WHERE $ep IN coalesce(new.episodes, []) AND new.uuid <> $old
                WITH old, collect(new.uuid)[0] AS new_uuid
                SET old.contradicted_by = new_uuid,
                    old.contradicted_at = datetime(),
                    old.contradicted_by_episode = $ep
                """,
                old=contradicted_uuid, ep=episode_uuid,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("could not persist contradiction link (%s)", str(exc)[:120])

    async def _revert_unjustified_invalidations(
        self, *, since: datetime, new_facts: list[str], own_edge_uuids: list[str],
        new_edges: list[tuple[str, str, str | None]] | None = None,
    ) -> list[str]:
        """Restore edges this write invalidated without a credible reason. Returns their uuids.

        Graphiti's ``add_episode`` retires edges an LLM judged contradicted by the new episode.
        That judgement over-reaches (see ``invalidation_is_credible`` for the measurement), and it
        fails SILENTLY: no error, no warning, and the write reports success. So every write audits
        its own collateral damage and puts back what it cannot justify.

        The test is deliberately one-sided, and has two halves. An invalidation stands only if some
        edge this write created could STRUCTURALLY have replaced the retired one (``could_replace``
        — same entity pair, or same endpoint plus relation name), AND some fact it extracted is
        about the same subject while differing by a value (``invalidation_is_credible``). Erring
        this way keeps a fact that perhaps should have been retired, which the contradiction review
        queue can still catch, instead of losing one silently (R8).

        The structural half was added 2026-08-19 after measuring the lexical half against the live
        graph and finding it far too permissive: of 526 retired edges, it had reverted 11 (2%),
        while applying the structural test retrospectively would have blocked **75.6%** of the 479
        retirements whose originating write could be identified (stable at 74-84% across join
        windows of 5-60 seconds). An independent report on a different backend and a different
        extraction model — getzep/graphiti#1728 — hand-audited 4 retirements and found 3 wrong.
        A veto that agrees with reality 2% of the time is decoration; this is the correction.

        Nothing is destroyed by the revert either: the ``invalid_at`` being cleared is preserved in
        ``invalidation_reverted_from`` so the original decision stays inspectable (R4). Timestamps
        are stored as ISO **strings** — a datetime-valued custom property breaks Graphiti's later
        dedupe prompts for the whole scope, and this writes to edges on every single write.

        Best-effort: the write has already succeeded and must never be undone by a failure here.
        """
        driver = self._driver()
        if driver is None or not new_facts:
            # No facts extracted means nothing could have contradicted anything, but it also means
            # we have nothing to judge against — so we cannot second-guess Graphiti either way.
            return []
        try:
            found = await driver.execute_query(
                """
                MATCH ()-[e:RELATES_TO]->()
                WHERE e.expired_at IS NOT NULL AND e.expired_at >= datetime($since)
                  AND NOT e.uuid IN $own
                RETURN e.uuid AS uuid, e.fact AS fact, e.group_id AS scope,
                       toString(e.invalid_at) AS invalid_at,
                       e.source_node_uuid AS src, e.target_node_uuid AS dst, e.name AS name
                """,
                since=since.isoformat(), own=own_edge_uuids,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("could not audit invalidations (%s)", str(exc)[:120])
            return []

        def field(record, key):
            """Tolerate a record that lacks the key — the graph backend is a seam."""
            try:
                return record[key]
            except (KeyError, IndexError, TypeError):
                return None

        shapes = list(new_edges or [])
        rows = []
        for record in found.records:
            old = record["fact"] or ""
            retired_shape = (field(record, "src"), field(record, "dst"), field(record, "name"))
            # Structural first: it is deterministic, needs no text, and is the half that was
            # missing. Both sides must be known to judge — with no new-edge shapes, or a retired
            # edge whose endpoints we could not read, this abstains (True) and leaves the decision
            # to the lexical test. Abstaining keeps a fact; treating unknown as "no replacement
            # exists" would revert every invalidation the moment a query changed shape.
            judgeable = bool(shapes) and retired_shape[0] is not None and retired_shape[1] is not None
            structural_ok = (
                any(could_replace(shape, retired_shape) for shape in shapes) if judgeable else True
            )
            lexical_ok = any(invalidation_is_credible(old, new) for new in new_facts)
            if structural_ok and lexical_ok:
                continue
            if not structural_ok:
                reason = "no edge this write created could have replaced it (no shared entity pair or endpoint+relation)"
            else:
                closest = max((subject_overlap(old, new) for new in new_facts), default=0.0)
                reason = f"no extracted fact shares its subject (best overlap {closest:.2f})"
            rows.append({
                "uuid": record["uuid"],
                "from": record["invalid_at"],
                "reason": reason,
            })
            logger.warning(
                "reverting unjustified invalidation of %s in %s (%s): %r",
                record["uuid"], record["scope"], reason, old[:160],
            )

        if not rows:
            return []
        try:
            await driver.execute_query(
                """
                UNWIND $rows AS row
                MATCH ()-[e:RELATES_TO {uuid: row.uuid}]->()
                SET e.expired_at = NULL,
                    e.invalid_at = NULL,
                    e.invalidation_reverted_at = $now,
                    e.invalidation_reverted_from = row.from,
                    e.invalidation_reverted_reason = row.reason
                """,
                rows=rows, now=datetime.now(timezone.utc).isoformat(),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("could not revert invalidations (%s)", str(exc)[:160])
            return []
        return [row["uuid"] for row in rows]

    def _dedup_scopes(self, scope: str, override: list[str] | None = None) -> list[str]:
        # Dedup ONLY within the target scope (R7). Deduping a project write against
        # global discarded project-contextualized restatements ("Acme-API uses BigDecimal
        # for monetary/price values") that recall from the project's seat needs — the
        # global version ("BigDecimal for money in Java") ranks low for an Acme-API-phrased
        # query. Cross-scope redundancy is the curation engine's job (fact-to-fact dedup +
        # promotion candidates), not the write path's.
        #
        # An explicit `override` widens it for a deliberate caller. The consolidation engine needs
        # this: a promotion writes into a scope that is typically EMPTY (that is the point of
        # promoting), so scope-only dedup can never notice that the wider tier already holds the
        # same knowledge. The first real promotion did exactly that — see
        # ConsolidationEngine._promotion_dedup_scopes. Only the caller may widen it; the default
        # stays narrow for the reason above.
        return list(override) if override else [scope]

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

    async def _existing_hash(self, content_hash: str, scopes: list[str]) -> str | None:
        """UUID of an already-stored Episodic node with this exact content hash in any of *scopes*.

        Best-effort: any driver error (or no driver, as in unit tests) means "no exact match found" and
        the write falls through to the normal vector dedup — the hash guard never blocks a legitimate write.
        """
        driver = self._driver()
        if driver is None:
            return None
        try:
            result = await driver.execute_query(
                """
                MATCH (e:Episodic {content_hash: $h})
                WHERE e.group_id IN $scopes
                RETURN e.uuid AS uuid LIMIT 1
                """,
                h=content_hash, scopes=scopes,
            )
            records = result.records
            return records[0]["uuid"] if records else None
        except Exception as exc:  # noqa: BLE001 — hash guard is an optimization; never fail the write
            logger.debug("content-hash lookup failed (%s); skipping the exact-dup guard", str(exc)[:120])
            return None

    async def _stamp_episode(
        self, episode_uuid: str | None, content_hash: str, provenance: Provenance | None = None,
    ) -> None:
        """Persist content_hash + provenance on the stored Episodic node.

        The hash makes future re-submits hit the 4a guard; the provenance makes the write
        attributable (roadmap item 13) — which is only possible AT write time, never afterwards.
        Both go in one query because they are stamped on the same node at the same moment.
        """
        driver = self._driver()
        if driver is None or episode_uuid is None:
            return
        props = {"content_hash": content_hash}
        if provenance is not None:
            props.update(provenance.as_props())
        try:
            await driver.execute_query(
                "MATCH (e:Episodic {uuid: $uuid}) SET e += $props",
                uuid=episode_uuid, props=props,
            )
        except Exception as exc:  # noqa: BLE001 — a missed stamp only weakens future dedup/attribution
            logger.debug("could not stamp episode %s (%s)", episode_uuid, str(exc)[:120])

    async def _canonicalize_edge_names(self, result) -> None:
        """Rename stored edges whose name has a confident schema equivalent. Best-effort.

        Only renames where :func:`canonical_edge_name` is confident and direction-preserving; an
        unknown relation keeps its own name rather than being flattened into a generic one.
        """
        driver = self._driver()
        if driver is None:
            return
        renames = []
        for edge in getattr(result, "edges", []) or []:
            name = getattr(edge, "name", None)
            uuid = getattr(edge, "uuid", None)
            if not name or not uuid:
                continue
            canonical = canonical_edge_name(name)
            if canonical != name:
                renames.append({"uuid": uuid, "name": canonical, "was": name})
        if not renames:
            return
        try:
            await driver.execute_query(
                """
                UNWIND $renames AS r
                MATCH ()-[e:RELATES_TO {uuid: r.uuid}]->()
                SET e.name = r.name, e.name_before_canonicalization = r.was
                """,
                renames=renames,
            )
            logger.info(
                "canonicalized %d edge name(s): %s", len(renames),
                ", ".join(f"{r['was']}->{r['name']}" for r in renames[:5]),
            )
        except Exception as exc:  # noqa: BLE001 — cosmetic normalization must never fail a write
            logger.debug("edge-name canonicalization skipped (%s)", str(exc)[:120])

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


def better_scope_than_global(
    content: str, projects: list[str], cluster_resolver=None,
) -> str | None:
    """A more accurate scope than ``global`` for this content, or None to leave it global.

    Decided purely by which known projects the content NAMES:

    * exactly one  -> that project's scope. It is project knowledge, however it was labelled.
    * two or more in the SAME cluster -> that cluster. This is the common real case: the measured
      pollution included the same fact twice in global, once per trading project.
    * two or more across DIFFERENT clusters -> global is correct, allow it.
    * none named -> plausibly universal, allow it.

    Naming is deliberately the only signal. Anything cleverer (asking a model whether this is
    "really universal") would be a per-write LLM guess, and ``_resolve_scope`` already refuses to
    trust one of those for exactly this decision.

    Module-level and pure so the write-time GATE and the one-off MIGRATION of already-polluted
    facts (``scripts/refile_global_facts.py``) apply the identical rule. If they could drift, a
    migration would move facts the gate would then keep re-admitting.
    """
    named = [
        p for p in projects
        if p and re.search(rf"\b{re.escape(p)}\b", content or "", re.IGNORECASE)
    ]
    if not named:
        return None
    if len(named) == 1:
        return Scope.project(named[0])

    clusters = set()
    for pid in named:
        try:
            clusters.add(cluster_resolver(pid) if cluster_resolver else None)
        except Exception:  # noqa: BLE001 — a bad registry entry must not decide scope
            clusters.add(None)
    if len(clusters) == 1:
        only = next(iter(clusters))
        if only:
            return Scope.cluster(only)
    # Spans domains (or the projects have no shared cluster) — global is the honest home.
    return None


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

    from synapse.core.registry import all_projects, cluster_of

    return WritePipeline(
        graphiti=graphiti,
        embedder=build_embedder(),
        index=Neo4jVectorIndex(graphiti),
        triage=ClaudeTriage(settings.anthropic_api_key, settings.triage_model),
        known_projects=lambda: list(all_projects()),
        cluster_resolver=cluster_of,
    )
