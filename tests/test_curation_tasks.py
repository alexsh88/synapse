"""Celery curation tasks — unit tests for task bodies + registration smoke tests.

Task bodies (_scan_suggestions, _scan_health) are tested with fake engine/graphiti
and a fake Redis client injected via sys.modules patching. No live Neo4j or Redis
needed. Pattern mirrors test_write_pipeline.py and test_curation_engine.py.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from synapse.core.curation_engine import CurationSuggestions

UTC = timezone.utc


# ──────────────────────────────────────────────────────────────────────────────
# Registration smoke tests (preserved from the original file)
# ──────────────────────────────────────────────────────────────────────────────


def test_tasks_and_schedule_registered():
    from synapse.workers.celery_app import celery_app

    names = set(celery_app.tasks.keys())
    assert "synapse.curation.scan_suggestions" in names
    assert "synapse.curation.scan_health" in names

    schedule = celery_app.conf.beat_schedule
    assert schedule["nightly-curation-scan"]["task"] == "synapse.curation.scan_suggestions"
    assert schedule["nightly-health-scan"]["task"] == "synapse.curation.scan_health"


def test_broker_is_redis():
    from synapse.workers.celery_app import celery_app

    assert celery_app.conf.broker_url.startswith("redis://")


# ──────────────────────────────────────────────────────────────────────────────
# Helpers / fakes
# ──────────────────────────────────────────────────────────────────────────────


def _make_fake_suggestions(**overrides) -> CurationSuggestions:
    """Minimal CurationSuggestions with no real data (task only reads counts)."""
    return CurationSuggestions(
        duplicates=overrides.get("duplicates", []),
        stale=overrides.get("stale", []),
        review_pairs=overrides.get("review_pairs", []),
        generated_at=datetime.now(UTC),
    )


class FakeGraphHealth:
    """Matches the fields _scan_health extracts."""
    total_nodes: int = 42
    superseded_edges: int = 7

    def model_dump_json(self) -> str:
        import json
        return json.dumps({"total_nodes": self.total_nodes, "superseded_edges": self.superseded_edges})


class FakeGraphiti:
    """Minimal stand-in for the Graphiti object (only .close() is needed)."""
    def __init__(self):
        self.closed = False

    async def close(self):
        self.closed = True


class FakeCurationEngine:
    def __init__(self, suggestions: CurationSuggestions):
        self._suggestions = suggestions

    async def suggestions(self, scopes=None):
        return self._suggestions


class FakeGraphService:
    def __init__(self, health):
        self._health = health

    async def health(self):
        return self._health


# ──────────────────────────────────────────────────────────────────────────────
# _scan_suggestions tests
# ──────────────────────────────────────────────────────────────────────────────


async def test_scan_suggestions_returns_correct_counts():
    """_scan_suggestions returns a summary dict with the right counts and caches the JSON."""
    from synapse.core.curation_engine import DuplicateCluster, FactRef, StaleItem

    dupes = [
        DuplicateCluster(
            scope="project_acme-api",
            canonical=FactRef(uuid="e1", fact="fact a"),
            duplicates=[FactRef(uuid="e2", fact="fact b")],
            max_similarity=0.98,
        )
    ]
    stale = [StaleItem(uuid="e3", fact="old fact", scope="global")]
    suggestions = _make_fake_suggestions(duplicates=dupes, stale=stale)

    fake_graphiti = FakeGraphiti()
    fake_engine = FakeCurationEngine(suggestions)
    cached: dict[str, str] = {}

    def fake_cache(key: str, value: str) -> None:
        cached[key] = value

    with (
        patch("synapse.workers.curation_tasks.build_graphiti", return_value=fake_graphiti),
        patch("synapse.workers.curation_tasks.build_curation_engine", return_value=fake_engine),
        patch("synapse.workers.curation_tasks._cache", side_effect=fake_cache),
    ):
        from synapse.workers.curation_tasks import _scan_suggestions

        result = await _scan_suggestions()

    assert result["duplicates"] == 1
    assert result["stale"] == 1
    assert result["review_pairs"] == 0
    # Graphiti must be closed in the finally block even on success.
    assert fake_graphiti.closed


async def test_scan_suggestions_closes_graphiti_on_exception():
    """_scan_suggestions closes Graphiti even when the engine raises."""
    fake_graphiti = FakeGraphiti()

    class _ExplodingEngine:
        async def suggestions(self, scopes=None):
            raise RuntimeError("neo4j down")

    with (
        patch("synapse.workers.curation_tasks.build_graphiti", return_value=fake_graphiti),
        patch("synapse.workers.curation_tasks.build_curation_engine", return_value=_ExplodingEngine()),
        patch("synapse.workers.curation_tasks._cache"),
    ):
        from synapse.workers.curation_tasks import _scan_suggestions

        with pytest.raises(RuntimeError, match="neo4j down"):
            await _scan_suggestions()

    assert fake_graphiti.closed


async def test_scan_suggestions_caches_json():
    """_scan_suggestions caches the suggestions JSON under the correct key."""
    from synapse.workers.curation_tasks import SUGGESTIONS_KEY

    suggestions = _make_fake_suggestions()
    fake_graphiti = FakeGraphiti()
    fake_engine = FakeCurationEngine(suggestions)
    cached: dict[str, str] = {}

    def fake_cache(key: str, value: str) -> None:
        cached[key] = value

    with (
        patch("synapse.workers.curation_tasks.build_graphiti", return_value=fake_graphiti),
        patch("synapse.workers.curation_tasks.build_curation_engine", return_value=fake_engine),
        patch("synapse.workers.curation_tasks._cache", side_effect=fake_cache),
    ):
        from synapse.workers.curation_tasks import _scan_suggestions

        await _scan_suggestions()

    assert SUGGESTIONS_KEY in cached
    import json
    parsed = json.loads(cached[SUGGESTIONS_KEY])
    assert "duplicates" in parsed


# ──────────────────────────────────────────────────────────────────────────────
# _scan_health tests
# ──────────────────────────────────────────────────────────────────────────────


async def test_scan_health_returns_correct_fields():
    """_scan_health returns total_nodes and superseded_edges from GraphService.health()."""
    health = FakeGraphHealth()
    fake_graphiti = FakeGraphiti()
    fake_graph_service = FakeGraphService(health)
    cached: dict[str, str] = {}

    def fake_cache(key: str, value: str) -> None:
        cached[key] = value

    import synapse.workers.curation_tasks as _mod

    # GraphService is imported inside _scan_health via a local import, so we intercept
    # via sys.modules to ensure the local import resolves to our fake.
    fake_gs_module = MagicMock()
    fake_gs_module.GraphService = MagicMock(return_value=fake_graph_service)

    with (
        patch.object(_mod, "build_graphiti", return_value=fake_graphiti),
        patch.object(_mod, "_cache", side_effect=fake_cache),
        patch.dict(sys.modules, {"synapse.core.graph_queries": fake_gs_module}),
    ):
        result = await _mod._scan_health()

    assert result["total_nodes"] == 42
    assert result["superseded_edges"] == 7


async def test_scan_health_closes_graphiti_on_exception():
    """_scan_health closes Graphiti even when GraphService.health() raises."""
    fake_graphiti = FakeGraphiti()

    class _ExplodingGraphService:
        async def health(self):
            raise RuntimeError("health check failed")

    # Import the module first to ensure it's loaded; then patch the names
    # it actually resolves at call-time (not a reloaded copy).
    import synapse.workers.curation_tasks as _mod

    fake_gs_module = MagicMock()
    fake_gs_module.GraphService = MagicMock(return_value=_ExplodingGraphService())

    with (
        patch.object(_mod, "build_graphiti", return_value=fake_graphiti),
        patch.dict(sys.modules, {"synapse.core.graph_queries": fake_gs_module}),
    ):
        with pytest.raises(RuntimeError, match="health check failed"):
            await _mod._scan_health()

    assert fake_graphiti.closed


# ──────────────────────────────────────────────────────────────────────────────
# Redis connection reuse (_cache / _get_redis_client)
# ──────────────────────────────────────────────────────────────────────────────


def test_cache_reuses_redis_client():
    """_get_redis_client returns the same client on repeated calls (connection reuse)."""
    import importlib
    import synapse.workers.curation_tasks as _mod

    # Reset module-level singleton so the test is isolated.
    _mod._redis_client = None

    fake_client = MagicMock()
    with patch("synapse.workers.curation_tasks.redis") as mock_redis_module:
        mock_redis_module.from_url.return_value = fake_client

        c1 = _mod._get_redis_client()
        c2 = _mod._get_redis_client()

    # from_url called exactly once — client is reused.
    assert mock_redis_module.from_url.call_count == 1
    assert c1 is c2
    # Clean up singleton.
    _mod._redis_client = None


def test_cache_swallows_redis_errors():
    """_cache logs a warning and does NOT raise when Redis is unavailable."""
    import importlib
    import synapse.workers.curation_tasks as _mod

    _mod._redis_client = None

    with patch("synapse.workers.curation_tasks.redis") as mock_redis_module:
        mock_client = MagicMock()
        mock_client.set.side_effect = ConnectionError("redis down")
        mock_redis_module.from_url.return_value = mock_client

        # Must not raise — caching is best-effort.
        _mod._cache("some:key", '{"x": 1}')

    _mod._redis_client = None
