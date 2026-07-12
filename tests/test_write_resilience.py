"""Tests for Ollama-outage write resilience (queue-and-replay).

All tests use fakes/mocks — no real Ollama, Neo4j, or Redis.

Items covered:
  1. Embedder-down → PendingCapture created, WriteResult.degraded=True
  2. Neo4j also down → hard fail, no infinite loop
  3. Replay task success path
  4. Replay task failure increments retry_count
  5. Replay task gives up at 5 attempts
  6. FACT_VECTOR_INDEX importable from both write_pipeline and curation_engine
"""

from __future__ import annotations

import pytest

from synapse.core.write_pipeline import (
    NearestFact,
    Outcome,
    TriageVerdict,
    WritePipeline,
    _is_embedder_down,
)

# ──────────────────────────────────────────────────────────────────────────────
# Fakes shared across tests
# ──────────────────────────────────────────────────────────────────────────────

STORABLE = TriageVerdict(
    worth_storing=True, knowledge_type="decision", is_global=False, confidence=0.8
)


class FakeTriage:
    def __init__(self, verdict: TriageVerdict):
        self._verdict = verdict

    async def classify(self, content, hint_type):
        return self._verdict

    async def adjudicate(self, new_content, existing_fact):
        from synapse.core.write_pipeline import Adjudication, Relation
        return Adjudication(relation=Relation.DISTINCT)


class FakeIndex:
    def __init__(self, nearest: NearestFact | None = None):
        self._nearest = nearest

    async def nearest(self, vec, scopes):
        return self._nearest


class _NoRecords:
    records: list = []


class _RecordWithUUID:
    def __init__(self, uuid: str):
        self.records = [{"uuid": uuid}]


# ──────────────────────────────────────────────────────────────────────────────
# Item 1: embedder down → PendingCapture queued, WriteResult.degraded=True
# ──────────────────────────────────────────────────────────────────────────────


class _DownEmbedder:
    """Embedder that always raises ConnectionRefusedError."""

    async def create(self, input_data: str) -> list[float]:
        raise ConnectionRefusedError("Connection refused")


class _RecordingDriver:
    """Driver that records Cypher queries; hash-lookup always returns no match."""

    def __init__(self):
        self.queries: list[str] = []
        self.params: list[dict] = []

    async def execute_query(self, query: str, **kwargs):
        self.queries.append(query)
        self.params.append(kwargs)
        return _NoRecords()


class _RecordingGraphiti:
    def __init__(self, driver: _RecordingDriver):
        self.driver = driver
        self.add_episode_calls: list[dict] = []

    async def add_episode(self, **kwargs):
        self.add_episode_calls.append(kwargs)
        raise RuntimeError("should not be reached when embedder is down")


async def test_embedder_down_queues_pending_capture():
    driver = _RecordingDriver()
    graphiti = _RecordingGraphiti(driver)
    pipeline = WritePipeline(
        graphiti=graphiti,
        embedder=_DownEmbedder(),
        index=FakeIndex(None),
        triage=FakeTriage(STORABLE),
        dedup_threshold=0.9,
        relate_floor=0.75,
    )
    # Disable health probe (tested separately; would also fail here without httpx mock)
    pipeline._health_checked = True

    result = await pipeline.remember(
        "Decided to use BGE-M3 for embeddings in Synapse.", project_id="synapse"
    )

    assert result.degraded is True
    assert "queued" in result.reason
    # A PendingCapture MERGE should have been issued on the driver.
    assert any("PendingCapture" in q for q in driver.queries), (
        f"Expected PendingCapture write in driver queries, got: {driver.queries}"
    )
    assert any("pending_replay" in q for q in driver.queries)
    # add_episode must NOT have been called (embedder never reached that far).
    assert graphiti.add_episode_calls == []


# ──────────────────────────────────────────────────────────────────────────────
# Item 2: Neo4j also down → hard fail, no infinite loop
# ──────────────────────────────────────────────────────────────────────────────


class _DownDriver:
    """Driver that always raises (simulates Neo4j being down)."""

    async def execute_query(self, query: str, **kwargs):
        raise OSError("Neo4j unavailable")


class _DownDriverGraphiti:
    def __init__(self):
        self.driver = _DownDriver()

    async def add_episode(self, **kwargs):
        raise OSError("Neo4j unavailable")


async def test_neo4j_also_down_hard_fails():
    graphiti = _DownDriverGraphiti()
    pipeline = WritePipeline(
        graphiti=graphiti,
        embedder=_DownEmbedder(),
        index=FakeIndex(None),
        triage=FakeTriage(STORABLE),
        dedup_threshold=0.9,
        relate_floor=0.75,
    )
    pipeline._health_checked = True

    # When the embedder is down AND Neo4j is down, the _queue_for_replay call
    # raises from execute_query (Neo4j down). That propagates up — hard failure.
    with pytest.raises(Exception):
        await pipeline.remember(
            "Decided to use BGE-M3 for embeddings in Synapse.", project_id="synapse"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Items 3-5: replay task logic
# ──────────────────────────────────────────────────────────────────────────────


class _ReplayDriver:
    """Driver with injectable pending records and query log."""

    def __init__(self, pending_records: list[dict]):
        self._pending = pending_records
        self.queries: list[str] = []
        self.params: list[dict] = []

    async def execute_query(self, query: str, **kwargs):
        self.queries.append(query)
        self.params.append(dict(kwargs))
        if "MATCH (p:PendingCapture {status: 'pending_replay'})" in query:
            return _ListRecords(self._pending)
        return _NoRecords()

    def get_set_params(self, expected_key: str):
        """Return all params from SET queries that contain the expected key."""
        return [p for p in self.params if expected_key in p]


class _ListRecords:
    def __init__(self, records: list[dict]):
        self.records = records


_PENDING_NODE = {
    "uuid": "pc-001",
    "hash": "abc123",
    "content": "Decided to use BGE-M3 for embeddings.",
    "type": "decision",
    "project_id": "project_synapse",
    "retry_count": 0,
}


class _SuccessPipeline:
    """Fake pipeline whose remember() always succeeds (non-degraded)."""

    async def remember(self, content, *, knowledge_type=None, project_id=None, force=False):
        from synapse.core.write_pipeline import WriteResult

        return WriteResult(
            outcome=Outcome.STORED,
            reason="stored",
            knowledge_type=knowledge_type,
            scope=f"project_{project_id}" if project_id else "global",
        )


class _FailPipeline:
    """Fake pipeline whose remember() always raises."""

    async def remember(self, content, *, knowledge_type=None, project_id=None, force=False):
        raise RuntimeError("embedder still down")


async def test_replay_task_success_path():
    from synapse.workers.replay_tasks import _run_replay

    driver = _ReplayDriver([dict(_PENDING_NODE)])
    summary = await _run_replay(driver, _SuccessPipeline())

    assert summary["replayed"] == 1
    assert summary["failed"] == 0
    assert summary["gave_up"] == 0
    # The driver must have received a SET status='replayed' query.
    assert any("'replayed'" in q for q in driver.queries), (
        f"Expected 'replayed' SET in queries: {driver.queries}"
    )


async def test_replay_task_failure_increments_retry():
    from synapse.workers.replay_tasks import _run_replay

    node = dict(_PENDING_NODE, retry_count=1)
    driver = _ReplayDriver([node])
    summary = await _run_replay(driver, _FailPipeline())

    assert summary["replayed"] == 0
    assert summary["failed"] == 1
    assert summary["gave_up"] == 0
    # retry_count should have been incremented to 2.
    rc_params = [p for p in driver.params if "rc" in p]
    assert rc_params, "Expected retry_count update query"
    assert rc_params[-1]["rc"] == 2


async def test_replay_task_gives_up_at_5():
    from synapse.workers.replay_tasks import _run_replay

    # retry_count=4 → next failure brings it to 5 → give up.
    node = dict(_PENDING_NODE, retry_count=4)
    driver = _ReplayDriver([node])
    summary = await _run_replay(driver, _FailPipeline())

    assert summary["gave_up"] == 1
    assert summary["replayed"] == 0
    assert summary["failed"] == 0
    # status should be set to 'replay_failed'.
    assert any("'replay_failed'" in q for q in driver.queries), (
        f"Expected 'replay_failed' in queries: {driver.queries}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Item 6: FACT_VECTOR_INDEX constant importable
# ──────────────────────────────────────────────────────────────────────────────


def test_vector_index_constant_importable_from_module():
    from synapse.core.vector_index import FACT_VECTOR_INDEX
    assert FACT_VECTOR_INDEX == "synapse_relates_fact_vec"


def test_vector_index_constant_used_in_write_pipeline():
    # write_pipeline now imports and uses the shared constant (not a local string literal).
    import synapse.core.write_pipeline as wp

    # The module should NOT define its own _FACT_VECTOR_INDEX string literal any more.
    assert not hasattr(wp, "_FACT_VECTOR_INDEX"), (
        "write_pipeline still defines _FACT_VECTOR_INDEX locally — should import FACT_VECTOR_INDEX"
    )
    # But it must expose the imported symbol.
    assert hasattr(wp, "FACT_VECTOR_INDEX")
    assert wp.FACT_VECTOR_INDEX == "synapse_relates_fact_vec"


def test_vector_index_constant_used_in_curation_engine():
    from synapse.core.curation_engine import _FACT_VECTOR_INDEX
    from synapse.core.vector_index import FACT_VECTOR_INDEX

    assert _FACT_VECTOR_INDEX == FACT_VECTOR_INDEX


# ──────────────────────────────────────────────────────────────────────────────
# _is_embedder_down helper
# ──────────────────────────────────────────────────────────────────────────────


def test_is_embedder_down_connection_refused():
    assert _is_embedder_down(ConnectionRefusedError())


def test_is_embedder_down_os_error_econnrefused():
    import errno as _errno
    exc = OSError(_errno.ECONNREFUSED, "Connection refused")
    assert _is_embedder_down(exc)


def test_is_embedder_down_message_match():
    assert _is_embedder_down(RuntimeError("connection refused to host"))
    assert _is_embedder_down(RuntimeError("Could not connect to ollama server"))


def test_is_embedder_down_does_not_catch_logic_errors():
    assert not _is_embedder_down(ValueError("bad schema"))
    assert not _is_embedder_down(RuntimeError("extraction failed: unexpected token"))
    assert not _is_embedder_down(KeyError("missing field"))
