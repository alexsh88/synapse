"""Temporal invariant tests for Synapse's knowledge engine (WP-C item 6).

All tests use inline fakes — no live Neo4j, no Redis, no Anthropic API.
Tests verify the four core temporal invariants:

1. Concurrent remembers produce no shared state (no scope/group_id cross-contamination).
2. Point-in-time temporal filter: as_of=t1 sees A (not B); as_of=now sees B (not A).
3. forget() uses COALESCE — re-forgetting does NOT move the original invalidation time.
4. update() failure on the new-store step leaves the old fact valid (no invalidation issued).
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

UTC = timezone.utc


# ──────────────────────────────────────────────────────────────────────────────
# Fakes
# ──────────────────────────────────────────────────────────────────────────────


class _FakeEmbedder:
    async def create(self, text: str) -> list[float]:
        return [0.1] * 1024


class _FakeIndex:
    async def nearest(self, vec, scopes):
        return None


class _Episode:
    uuid = "ep-fake"


class _Node:
    def __init__(self, name):
        self.name = name


class _Edge:
    def __init__(self, name, fact):
        self.name = name
        self.fact = fact


class _AddResult:
    episode = _Episode()
    nodes = [_Node("FakeEntity")]
    edges = [_Edge("related_to", "FakeEntity related_to Other")]


class _FakeGraphiti:
    """Minimal Graphiti stand-in for the write pipeline (no driver, no Neo4j)."""

    def __init__(self):
        self.calls: list[dict] = []

    async def add_episode(self, **kwargs):
        self.calls.append(kwargs)
        return _AddResult()


class _FakeTriage:
    from synapse.core.write_pipeline import Adjudication, Relation, TriageVerdict  # noqa: F401

    async def classify(self, content, hint_type):
        from synapse.core.write_pipeline import TriageVerdict
        return TriageVerdict(worth_storing=True, knowledge_type="decision", is_global=False, confidence=0.9)

    async def adjudicate(self, new_content, existing_fact):
        from synapse.core.write_pipeline import Adjudication, Relation
        return Adjudication(relation=Relation.DISTINCT)


class _CapturingDriver:
    """Records every Cypher query issued (used to inspect what the engine emits)."""

    def __init__(self):
        self.queries: list[tuple[str, dict]] = []

    async def execute_query(self, query: str, **params):
        self.queries.append((query, params))

        class _R:
            records = [{"uuid": params.get("id", "fake-uuid")}]

        return _R()


class _CapturingDriverNoResult:
    """Like _CapturingDriver but execute_query returns empty records (simulates not-found)."""

    def __init__(self):
        self.queries: list[tuple[str, dict]] = []

    async def execute_query(self, query: str, **params):
        self.queries.append((query, params))

        class _R:
            records: list = []

        return _R()


class _CapturingGraphiti:
    """Wraps a _CapturingDriver so KnowledgeEngine.forget() can use it."""

    def __init__(self, driver):
        self.driver = driver


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _make_pipeline(graphiti=None):
    from synapse.core.write_pipeline import WritePipeline

    fg = graphiti or _FakeGraphiti()
    return WritePipeline(
        graphiti=fg,
        embedder=_FakeEmbedder(),
        index=_FakeIndex(),
        triage=_FakeTriage(),
        dedup_threshold=0.9,
        relate_floor=0.75,
    ), fg


# ──────────────────────────────────────────────────────────────────────────────
# Test 1: concurrent remembers — no shared state
# ──────────────────────────────────────────────────────────────────────────────


async def test_concurrent_remembers_no_shared_state():
    """10 concurrent remember() calls each produce an independent, correctly-scoped store.

    Asserts:
    - Exactly 10 add_episode calls are made.
    - No two calls share a group_id from a different call's project (no cross-contamination).
    - Each call's group_id matches its own project_id exactly.
    """
    pipeline, fg = _make_pipeline()

    project_ids = [f"project_{i}" for i in range(10)]
    contents = [f"Decision number {i}: we chose approach {i}." for i in range(10)]

    results = await asyncio.gather(*(
        pipeline.remember(content, project_id=f"project_{i}")
        for i, content in enumerate(contents)
    ))

    # All 10 stores succeeded.
    assert len(fg.calls) == 10

    # Each result is scoped to its own project — no leakage.
    for i, result in enumerate(results):
        assert result.scope == f"project_project_{i}", (
            f"result {i} has wrong scope: {result.scope!r}"
        )

    # Each add_episode call carried the correct group_id.
    actual_group_ids = {call["group_id"] for call in fg.calls}
    expected_group_ids = {f"project_project_{i}" for i in range(10)}
    assert actual_group_ids == expected_group_ids, (
        f"group_id cross-contamination detected. Expected {expected_group_ids}, got {actual_group_ids}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Test 2: point-in-time temporal filter
# ──────────────────────────────────────────────────────────────────────────────


def test_supersede_point_in_time():
    """Fact A valid [t1, t2); Fact B valid [t2, now). temporal_filter must see the right fact."""
    from synapse.core.retrieval_engine import Fact, temporal_filter

    t1 = datetime(2026, 1, 1, tzinfo=UTC)
    t2 = datetime(2026, 6, 1, tzinfo=UTC)
    now = datetime(2026, 7, 11, tzinfo=UTC)

    # Fact A: valid from t1, superseded at t2.
    fact_a = Fact(uuid="a", fact="original fact", group_id="global", valid_at=t1, invalid_at=t2)
    # Fact B: valid from t2, still active.
    fact_b = Fact(uuid="b", fact="updated fact", group_id="global", valid_at=t2, invalid_at=None)

    facts = [fact_a, fact_b]

    # As-of t1 (in the middle of A's window, before B): only A.
    as_of_t1 = datetime(2026, 3, 15, tzinfo=UTC)  # t1 < as_of_t1 < t2
    at_t1 = temporal_filter(facts, as_of_t1)
    assert len(at_t1) == 1, f"Expected 1 fact at t1 window, got {len(at_t1)}: {at_t1}"
    assert at_t1[0].uuid == "a", f"Expected fact A at t1, got {at_t1[0].uuid}"

    # As-of now (after t2): only B.
    at_now = temporal_filter(facts, now)
    assert len(at_now) == 1, f"Expected 1 fact at now, got {len(at_now)}: {at_now}"
    assert at_now[0].uuid == "b", f"Expected fact B at now, got {at_now[0].uuid}"

    # As-of exactly t2 (B becomes valid, A expires): only B (invalid_at is exclusive upper bound).
    at_t2 = temporal_filter(facts, t2)
    # A's invalid_at == t2 so A is filtered out (invalid_at <= as_of); B's valid_at == t2 so B is kept.
    assert len(at_t2) == 1
    assert at_t2[0].uuid == "b"


# ──────────────────────────────────────────────────────────────────────────────
# Test 3: forget() twice — COALESCE preserves original invalidation timestamp
# ──────────────────────────────────────────────────────────────────────────────


async def test_forget_twice_preserves_original_invalidation():
    """Re-calling forget() on an already-invalidated fact must NOT move invalid_at.

    Verified by inspecting the Cypher query: it must contain COALESCE around
    e.invalid_at, not an unconditional SET e.invalid_at = datetime().
    """
    driver = _CapturingDriver()
    graphiti = _CapturingGraphiti(driver)

    from synapse.core.knowledge_engine import KnowledgeEngine

    engine = KnowledgeEngine(graphiti=graphiti)  # type: ignore[arg-type]

    # First forget.
    await engine.forget("fact-uuid-1", reason="first forget")
    # Second forget (simulating re-forgetting an already-invalid fact).
    await engine.forget("fact-uuid-1", reason="second forget")

    assert len(driver.queries) == 2

    for query, params in driver.queries:
        # The query MUST use COALESCE for invalid_at, not a bare SET.
        assert "coalesce(e.invalid_at" in query.lower() or "coalesce(e.invalid_at" in query, (
            f"forget() query missing COALESCE for e.invalid_at:\n{query}"
        )
        # It must NOT set invalid_at unconditionally.
        # (A bare "e.invalid_at = datetime()" without coalesce would corrupt history.)
        bare_set = "set e.invalid_at = datetime()"
        assert bare_set not in query.lower(), (
            f"forget() query contains unconditional SET e.invalid_at = datetime():\n{query}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Test 4: update() failure on new-store leaves old fact valid
# ──────────────────────────────────────────────────────────────────────────────


async def test_update_failure_leaves_old_fact_valid():
    """When remember() (new-store) raises, update() must NOT issue a forget() query.

    Asserts:
    - Result dict has success=False.
    - No Cypher query touching e.invalid_at was issued to the driver.
    """
    driver = _CapturingDriver()
    graphiti_obj = _CapturingGraphiti(driver)

    class _ExplodingPipeline:
        """Stand-in for WritePipeline whose remember() always raises."""

        async def remember(self, content, **kwargs):
            raise RuntimeError("simulated extraction failure")

    from synapse.core.knowledge_engine import KnowledgeEngine

    engine = KnowledgeEngine(graphiti=graphiti_obj)  # type: ignore[arg-type]
    # Inject the exploding writer so remember() fails without touching the driver.
    engine.writer = _ExplodingPipeline()

    result = await engine.update("old-fact-uuid", {"content": "new content"})

    assert result["success"] is False, f"Expected success=False, got: {result}"
    assert "error" in result

    # The critical assertion: no invalidation query was issued to the driver.
    invalidation_queries = [
        (q, p) for q, p in driver.queries
        if "invalid_at" in q.lower()
    ]
    assert invalidation_queries == [], (
        f"update() issued invalidation queries despite new-store failure: {invalidation_queries}"
    )
