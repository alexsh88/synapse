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

from openai import AsyncOpenAI

from graphiti_core import Graphiti
from graphiti_core.cross_encoder.client import CrossEncoderClient
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.llm_client.anthropic_client import AnthropicClient
from graphiti_core.llm_client.config import LLMConfig

from synapse.config import settings

logger = logging.getLogger("synapse.engine")

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
        self.writer = None
        self.reader = None
        self.graph = None  # GraphService — set in connect()
        self.curation = None  # CurationEngine — set in connect()
        self.capture = None  # CaptureEngine — set in connect()

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
        return self

    async def remember(self, content: str, **kwargs):
        result = await self.writer.remember(content, **kwargs)
        # Keep briefs fresh: a real write to a project busts that project's brief cache.
        if result.scope and result.scope.startswith("project_") and result.outcome.value in (
            "stored",
            "contradiction",
        ):
            await self.reader.invalidate_brief(result.scope.removeprefix("project_"))
        return result

    async def recall(self, query: str, **kwargs):
        return await self.reader.recall(query, **kwargs)

    async def brief(self, project_id: str, **kwargs):
        return await self.reader.brief(project_id, **kwargs)

    async def search(self, query: str, *, group_ids=None, limit: int = 10, as_of=None):
        """Search across all knowledge (group_ids=None) or a specific scope set."""
        return await self.reader.search(query, group_ids=group_ids, limit=limit, as_of=as_of)

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
                e.forget_reason = coalesce(e.forget_reason, $reason)
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
