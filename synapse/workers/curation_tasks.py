"""Celery curation tasks — read-only analysis cached to Redis (Phase 10).

These run on the beat schedule (see ``celery_app``). They build a short-lived
Graphiti connection, run the engine's **read-only** analysis, and cache the JSON
result under a Redis key the API can serve instantly. No task mutates the graph.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

import redis

from synapse.config import settings
from synapse.core.curation_engine import build_curation_engine
from synapse.core.knowledge_engine import build_graphiti
from synapse.workers.celery_app import celery_app

if TYPE_CHECKING:
    import redis as _redis_types

logger = logging.getLogger("synapse.curation.tasks")

SUGGESTIONS_KEY = "curation:suggestions"
HEALTH_KEY = "curation:health"
CACHE_TTL = 60 * 60 * 26  # a bit over a day, so the nightly scan always overlaps

# Module-level lazily-created Redis client — reused across all _cache() calls to avoid
# opening a new connection on every scan result write (spec WP-C item 4).
_redis_client: "redis.Redis | None" = None


def _get_redis_client() -> "redis.Redis":
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.from_url(settings.redis_url)
    return _redis_client


def _cache(key: str, value: str) -> None:
    try:
        _get_redis_client().set(key, value, ex=CACHE_TTL)
    except Exception as exc:  # noqa: BLE001 — caching is best-effort, never fail the scan
        logger.warning("could not cache %s: %s", key, exc)


@celery_app.task(name="synapse.curation.scan_suggestions")
def scan_suggestions() -> dict:
    return asyncio.run(_scan_suggestions())


async def _scan_suggestions() -> dict:
    graphiti = build_graphiti()
    try:
        engine = build_curation_engine(graphiti)
        suggestions = await engine.suggestions()
        _cache(SUGGESTIONS_KEY, suggestions.model_dump_json())
        summary = {
            "duplicates": len(suggestions.duplicates),
            "stale": len(suggestions.stale),
            "review_pairs": len(suggestions.review_pairs),
        }
        logger.info("curation scan: %s", summary)
        return summary
    finally:
        await graphiti.close()


@celery_app.task(name="synapse.curation.scan_health")
def scan_health() -> dict:
    return asyncio.run(_scan_health())


async def _scan_health() -> dict:
    from synapse.core.graph_queries import GraphService

    graphiti = build_graphiti()
    try:
        health = await GraphService(graphiti).health()
        _cache(HEALTH_KEY, health.model_dump_json())
        return {"total_nodes": health.total_nodes, "superseded_edges": health.superseded_edges}
    finally:
        await graphiti.close()


@celery_app.task(name="synapse.curation.consolidate")
def consolidate() -> dict:
    """Nightly consolidation pass (research Wave 2) — PROPOSE ONLY, never mutates knowledge.

    Runs after the suggestion/health scans so it sees a fresh duplicate analysis. Everything it
    produces lands in the review inbox (``GET /api/v1/curation/proposals``); nothing is applied
    without a human. Uses no cloud LLM, so it costs nothing to run every night.
    """
    return asyncio.run(_consolidate())


async def _consolidate() -> dict:
    from synapse.core.consolidation_engine import build_consolidation_engine

    graphiti = build_graphiti()
    try:
        # No `remember` wired here on purpose: the worker must not be able to write knowledge.
        # Applying a promotion is a human action through the API, which has the write path.
        engine = build_consolidation_engine(graphiti)
        run = await engine.propose()
        summary = {
            "merges_proposed": run.merges_proposed,
            "promotions_proposed": run.promotions_proposed,
            "already_known": run.already_known,
        }
        logger.info("consolidation pass: %s", summary)
        return summary
    finally:
        await graphiti.close()
