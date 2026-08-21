"""Synapse knowledge engine — the Graphiti wrapper.

Builds a Graphiti instance wired to Synapse's LOCKED stack (CLAUDE.md §1):

- **Extraction LLM:** Claude Sonnet 4.6 via Graphiti's ``AnthropicClient`` — the
  only outbound API call in the pipeline.
- **Embedder:** BGE-M3 (1024-dim) served locally by Ollama, reached through
  Graphiti's OpenAI-compatible ``OpenAIEmbedder`` pointed at Ollama's ``/v1``.
- **Cross-encoder:** a passthrough for the MVP. Graphiti's built-in RRF /
  node-distance rerankers do the ranking; a local ``BGERerankerClient`` can be
  swapped in later. We deliberately avoid Graphiti's default OpenAI reranker so
  there is **no OpenAI dependency anywhere**.

Nothing here reads or writes OpenAI. The embedder/reranker stay 100% local.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, TypeVar

from openai import AsyncOpenAI

from graphiti_core import Graphiti
from graphiti_core.cross_encoder.client import CrossEncoderClient
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.llm_client.anthropic_client import AnthropicClient
from graphiti_core.llm_client.config import LLMConfig

from synapse.config import settings

# Type-only: the runtime imports stay inside connect() because write_pipeline imports THIS
# module, and a top-level import would close the cycle. TYPE_CHECKING never executes.
if TYPE_CHECKING:
    from synapse.core.consolidation_engine import ConsolidationEngine
    from synapse.core.curation_engine import CurationEngine
    from synapse.core.graph_queries import GraphService
    from synapse.core.retrieval_engine import RetrievalEngine
    from synapse.core.session_capture import CaptureEngine
    from synapse.core.write_pipeline import WritePipeline

logger = logging.getLogger("synapse.engine")

_C = TypeVar("_C")


def _require(component: _C | None, name: str) -> _C:
    """Return a component built by ``connect()``, or say plainly that connect() was skipped.

    Without this the failure is ``AttributeError: 'NoneType' object has no attribute 'remember'``
    from somewhere deep in a request, which names neither the component nor the missing step.
    """
    if component is None:
        raise RuntimeError(
            f"KnowledgeEngine.{name} is not available — `await engine.connect()` was never called"
        )
    return component

# Ollama's OpenAI-compatible endpoint comes from settings.ollama_base_url (host default
# 127.0.0.1; the dockerized API overrides to host.docker.internal). The api_key is required
# by the OpenAI client object but ignored by Ollama.
OLLAMA_DUMMY_KEY = "ollama"

EMBED_MODEL = "bge-m3"
EMBED_DIM = 1024  # LOCKED at first ingestion (CLAUDE.md §1). Changing => re-embed all.


class PassthroughCrossEncoder(CrossEncoderClient):
    """MVP no-op reranker: returns passages in input order with descending scores.

    Graphiti requires a cross-encoder at construction and otherwise defaults to an
    LLM-based OpenAI reranker (which would need an OpenAI key). For the MVP the
    actual ranking is handled by Graphiti's built-in RRF / node-distance rerankers,
    so this stays unused. Replace with a local ``BGERerankerClient`` for quality.
    """

    async def rank(self, query: str, passages: list[str]) -> list[tuple[str, float]]:
        n = len(passages)
        return [(p, 1.0 - (i / n) if n else 1.0) for i, p in enumerate(passages)]


def build_llm_client() -> AnthropicClient:
    """Claude Sonnet 4.6 for entity/relationship extraction."""
    config = LLMConfig(
        api_key=settings.anthropic_api_key,
        model=settings.extraction_model,
        # Pin the "small" model to Sonnet too, so no OpenAI model name leaks in.
        small_model=settings.extraction_model,
    )
    return AnthropicClient(config=config)


class RetryingOpenAIEmbedder(OpenAIEmbedder):
    """OpenAIEmbedder that retries transient failures from local Ollama/bge-m3.

    Under concurrent load the GPU occasionally returns a NaN embedding, which
    Ollama surfaces as a 500. These are transient — a short backoff and retry
    (when momentary load has eased) almost always succeeds. Persistent failures
    still raise so the caller (and MCP `_safe`) can report them.
    """

    _ATTEMPTS = 3

    async def create(self, input_data):
        return await self._retry(super().create, input_data)

    async def create_batch(self, input_data_list):
        """Embed a batch, falling back to per-item on failure.

        Under load bge-m3 occasionally returns a NaN for a *batch* (Ollama 500);
        single-item embeds are far more reliable. So on batch failure we retry,
        then embed each item individually. We never substitute a zero vector — a
        zero vector is invalid for ``vector.similarity.cosine`` and would poison
        dedup — so a genuinely unembeddable item raises (surfaced via MCP `_safe`).
        """
        try:
            return await self._retry(super().create_batch, input_data_list, attempts=2)
        except Exception as exc:  # noqa: BLE001
            logger.warning("batch embed failed (%s); falling back to per-item for %d items",
                           exc, len(input_data_list))
            return [await self._retry(super().create, item) for item in input_data_list]

    async def _retry(self, fn, arg, *, attempts: int | None = None):
        last: Exception | None = None
        n = attempts or self._ATTEMPTS
        for attempt in range(n):
            try:
                return await fn(arg)
            except Exception as exc:  # noqa: BLE001 - transient embedder errors
                last = exc
                preview = repr(arg)[:120]
                logger.warning("embedder attempt %d/%d failed (%s); input=%s", attempt + 1, n, exc, preview)
                await asyncio.sleep(0.5 * (attempt + 1))
        # attempts <= 0 is a misconfigured caller, not a transient embedder failure. Without this
        # guard the loop never runs and `raise last` raises None, masking the real mistake.
        if last is None:
            raise RuntimeError(f"embedder retry ran zero attempts (attempts={n})")
        raise last


def build_embedder() -> OpenAIEmbedder:
    """BGE-M3 (1024-dim) via Ollama's OpenAI-compatible embeddings endpoint."""
    base_url = settings.ollama_base_url
    config = OpenAIEmbedderConfig(
        embedding_model=EMBED_MODEL,
        embedding_dim=EMBED_DIM,
        base_url=base_url,
        api_key=OLLAMA_DUMMY_KEY,
    )
    client = AsyncOpenAI(base_url=base_url, api_key=OLLAMA_DUMMY_KEY)
    return RetryingOpenAIEmbedder(config=config, client=client)


def build_graphiti() -> Graphiti:
    """Construct a Graphiti instance wired to the locked Synapse stack.

    Caller is responsible for ``await graphiti.build_indices_and_constraints()``
    once, and ``await graphiti.close()`` on shutdown.

    The extraction LLM is chosen by ``settings.extraction_mode`` (cloud / hybrid / local) —
    see ``synapse/core/extraction_clients.py`` and ``docs/research/local-extraction-models.md``.
    """
    from synapse.core.extraction_clients import build_extraction_client  # lazy: avoids circular import

    return Graphiti(
        settings.neo4j_uri,
        settings.neo4j_user,
        settings.neo4j_password,
        llm_client=build_extraction_client(),
        embedder=build_embedder(),
        cross_encoder=PassthroughCrossEncoder(),
        # Cap concurrent LLM/embedding calls. Local bge-m3 on the GPU intermittently
        # emits a NaN embedding (-> Ollama 500) under heavy concurrency; a low limit
        # plus RetryingOpenAIEmbedder keeps writes reliable. (Graphiti's SEMAPHORE_LIMIT
        # env var is read at import, too late to set here — so we pass it directly.)
        max_coroutines=settings.graphiti_max_coroutines,
    )


class KnowledgeEngine:
    """Cohesive entry point to Synapse's brain.

    Owns one Graphiti instance and exposes the write (``remember``) and read
    (``recall`` / ``brief``) paths over it. This is what the MCP server (Phase 4)
    and API (Phase 5) drive. Write/read are kept coherent: a successful store
    invalidates the affected project's cached brief.

        async with KnowledgeEngine() as engine:
            await engine.remember("We chose X because Y", project_id="acme-store")
            hits = await engine.recall("why X?", project_id="acme-store")
            brief = await engine.brief("acme-store")
    """

    def __init__(self, graphiti: Graphiti | None = None, *, redis=None) -> None:
        self.graphiti = graphiti or build_graphiti()
        self._redis = redis
        # All six are built by connect(); until then they are None and every use site goes
        # through _require(), so "used before connect()" fails with that sentence instead of
        # an AttributeError on None.
        self.writer: WritePipeline | None = None
        self.reader: RetrievalEngine | None = None
        self.graph: GraphService | None = None
        self.curation: CurationEngine | None = None
        self.capture: CaptureEngine | None = None
        self.consolidation: ConsolidationEngine | None = None

    async def connect(self) -> "KnowledgeEngine":
        # Lazy imports avoid a circular dependency (write_pipeline imports this module).
        from synapse.core.curation_engine import build_curation_engine
        from synapse.core.graph_queries import GraphService
        from synapse.core.retrieval_engine import build_retrieval_engine
        from synapse.core.write_pipeline import build_write_pipeline

        await self.graphiti.build_indices_and_constraints()
        self.writer = build_write_pipeline(self.graphiti)
        self.reader = build_retrieval_engine(self.graphiti, redis=self._redis)
        self.graph = GraphService(self.graphiti)
        self.curation = build_curation_engine(self.graphiti)
        from synapse.core.session_capture import build_capture_engine
        self.capture = build_capture_engine(self.graphiti, self.remember)
        # Consolidation gets self.remember (not the raw pipeline) so an applied promotion goes
        # through triage, credential redaction, dedup AND busts the affected brief cache.
        from synapse.core.consolidation_engine import build_consolidation_engine
        self.consolidation = build_consolidation_engine(self.graphiti, remember=self.remember)
        return self

    async def remember(self, content: str, **kwargs):
        # Attribute every write, from every path (MCP, API, capture, seed, replay, consolidation),
        # at ONE place. Provenance can only be captured at write time — see
        # synapse/core/provenance.py — so a caller that forgets to pass it must not produce an
        # anonymous write. An explicit value always wins.
        from synapse.core.provenance import resolve as resolve_provenance

        if kwargs.get("provenance") is None:
            kwargs["provenance"] = resolve_provenance()
        result = await _require(self.writer, "writer").remember(content, **kwargs)
        # Keep briefs fresh: a real write to a project busts that project's brief cache.
        if result.scope and result.scope.startswith("project_") and result.outcome.value in (
            "stored",
            "contradiction",
        ):
            await _require(self.reader, "reader").invalidate_brief(
                result.scope.removeprefix("project_")
            )
        return result

    async def remember_runbook(
        self,
        name: str,
        steps: list[str],
        *,
        project_id: str | None = None,
        cluster: str | None = None,
        purpose: str | None = None,
        prerequisites: str | None = None,
        verified_at=None,
    ):
        """Store an ordered procedure (roadmap item 18).

        Two writes, deliberately, because they serve different masters:

        1. **The prose episode**, through the normal ``remember`` path — triage, credential
           redaction, dedup, provenance. This is what makes the runbook reachable by ``recall()``,
           which searches fact edges. Extraction will mangle the step order in this copy and that
           is expected; it is the index, not the source of truth.
        2. **The structured node**, via :class:`~synapse.core.runbooks.RunbookStore`. Ordered
           steps written as a property, with no model in the path that could reorder them.

        The prose goes first so that step 2 usually *adopts* the entity node extraction just
        created, leaving one node that is both connected to the graph and structurally correct.

        Returns the :class:`~synapse.core.runbooks.RunbookRecord`. A failure in the prose write is
        logged and swallowed — losing searchability is bad, losing the procedure is worse.
        """
        from synapse.core.runbooks import RunbookStore, normalize_steps, runbook_prose
        from synapse.core.schema import Scope

        steps = normalize_steps(steps)
        # Scope precedence matches the write pipeline's: cluster > project > global.
        if cluster:
            target = Scope.cluster(cluster)
        elif project_id:
            target = Scope.project(project_id)
        else:
            target = Scope.GLOBAL

        try:
            await self.remember(
                runbook_prose(name, steps, purpose, prerequisites),
                project_id=project_id,
                cluster=cluster,
                source="runbook",
            )
        except Exception:  # noqa: BLE001 — the index is best-effort; the procedure is not
            logger.warning(
                "runbook %r: prose episode failed, storing the structured node anyway "
                "(recall() will not surface it until it is rewritten)", name, exc_info=True,
            )

        record = await RunbookStore(self.graphiti).upsert(
            name=name, scope=target, steps=steps, purpose=purpose,
            prerequisites=prerequisites, verified_at=verified_at,
        )
        if target.startswith("project_"):
            await _require(self.reader, "reader").invalidate_brief(target.removeprefix("project_"))
        return record

    async def runbooks(self, project_id: str | None = None, *, limit: int = 20):
        """Runbooks visible from *project_id*'s seat (global + cluster + project)."""
        from synapse.core.runbooks import RunbookStore
        from synapse.core.schema import Scope

        cluster = self.reader._cluster_for(project_id) if self.reader else None
        scopes = Scope.compose(project_id, cluster=cluster) if project_id else [Scope.GLOBAL]
        return await RunbookStore(self.graphiti).list_for_scopes(scopes, limit=limit)

    async def recall(self, query: str, **kwargs):
        return await _require(self.reader, "reader").recall(query, **kwargs)

    async def brief(self, project_id: str, **kwargs):
        return await _require(self.reader, "reader").brief(project_id, **kwargs)

    async def search(self, query: str, *, group_ids=None, limit: int = 10, as_of=None):
        """Search across all knowledge (group_ids=None) or a specific scope set."""
        return await _require(self.reader, "reader").search(
            query, group_ids=group_ids, limit=limit, as_of=as_of
        )

    async def relate(self, from_id: str, to_id: str, relationship_type: str) -> dict:
        """Manually link two entity nodes with a typed edge.

        Structural only — the manual edge has no fact embedding, so it shapes the
        graph (and graph-traversal recall) but won't surface in semantic search.
        """
        res = await self.graphiti.driver.execute_query(
            """
            MATCH (a:Entity {uuid: $from_id}), (b:Entity {uuid: $to_id})
            MERGE (a)-[r:RELATES_TO {name: $rel}]->(b)
            ON CREATE SET r.uuid = randomUUID(), r.created_at = datetime(),
                          r.group_id = a.group_id, r.manual = true,
                          r.valid_at = datetime(),
                          r.episodes = [],  // EntityEdge requires a list here, else
                                            // Graphiti's search fails to hydrate the edge
                          r.fact = a.name + ' ' + $rel + ' ' + b.name
            RETURN r.uuid AS uuid, r.fact AS fact
            """,
            from_id=from_id, to_id=to_id, rel=relationship_type,
        )
        if not res.records:
            return {"success": False, "error": "one or both ids not found (expects entity node uuids)"}
        rec = res.records[0]
        return {"success": True, "edge_uuid": rec["uuid"], "fact": rec["fact"]}

    async def forget(self, knowledge_id: str, reason: str | None = None) -> dict:
        """Temporal end, NOT deletion (R4): mark a fact invalid as of now.

        Uses COALESCE so that re-forgetting an already-invalidated fact does NOT
        move its supersession timestamp — preserving point-in-time history (R4).
        """
        res = await self.graphiti.driver.execute_query(
            """
            MATCH ()-[e:RELATES_TO {uuid: $id}]->()
            SET e.invalid_at = coalesce(e.invalid_at, datetime()),
                e.expired_at = coalesce(e.expired_at, datetime()),
                e.forget_reason = coalesce(e.forget_reason, $reason),
                // Retrieval feedback (roadmap item 14): forgetting a fact is an explicit judgement
                // that it was wrong — the strongest quality signal available, and far more
                // trustworthy than any inference about whether a recalled fact was "used".
                e.corrected_n = coalesce(e.corrected_n, 0) + 1,
                e.last_corrected_at = datetime()
            RETURN e.uuid AS uuid
            """,
            id=knowledge_id, reason=reason or "forgotten",
        )
        if not res.records:
            return {"success": False, "error": "fact id not found (expects a fact uuid from recall/search)"}
        return {"success": True, "knowledge_id": knowledge_id, "note": "marked invalid (temporal end, not deleted)"}

    async def update(self, knowledge_id: str, changes, *, project_id: str | None = None) -> dict:
        """Create a temporal version (R4): store the new content FIRST, then invalidate the old.

        Order of operations (locked decision — CLAUDE.md R4):
        1. Store the new fact via remember().
        2. ONLY on success, invalidate the old fact via forget().
        3. On remember failure, return success=False leaving the old fact valid.

        A brief double-valid window is acceptable; losing knowledge is not.
        """
        content = changes.get("content") if isinstance(changes, dict) else str(changes)
        if not content:
            return {"success": False, "error": "update requires new 'content' in changes"}
        try:
            result = await self.remember(content, project_id=project_id, force=True)
        except Exception as exc:  # noqa: BLE001
            logger.warning("update: remember step failed for %s (%s) — old fact left valid", knowledge_id, exc)
            return {"success": False, "error": f"new-fact store failed: {exc}"}
        superseded = await self.forget(knowledge_id, reason="superseded by update")
        return {
            "success": True,
            "superseded": superseded.get("success", False),
            "new": {"outcome": result.outcome.value, "episode_uuid": result.episode_uuid, "scope": result.scope},
        }

    async def close(self) -> None:
        await self.graphiti.close()

    async def __aenter__(self) -> "KnowledgeEngine":
        return await self.connect()

    async def __aexit__(self, *exc) -> None:
        await self.close()
