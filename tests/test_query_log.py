"""Query telemetry — shape, privacy, fail-open behaviour, and that reads actually record."""

from __future__ import annotations

import json

import pytest

from synapse.core import query_log
from synapse.core.query_log import MAX_ENTRIES, TOP_N_RESULTS, build_record, read_all, record
from synapse.core.retrieval_engine import RetrievalEngine
from tests.test_retrieval_engine import NOW, FakeQueries, FakeSearcher, fact


class FakeRedis:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.lists: dict[str, list[str]] = {}
        self.trims: list[tuple[str, int, int]] = []

    async def lpush(self, key, value):
        if self.fail:
            raise ConnectionError("redis down")
        self.lists.setdefault(key, []).insert(0, value)

    async def ltrim(self, key, start, stop):
        self.trims.append((key, start, stop))
        self.lists[key] = self.lists.get(key, [])[start:stop + 1]

    async def lrange(self, key, start, stop):
        return self.lists.get(key, [])[start:stop + 1]


class _Result:
    def __init__(self, uuid, score=None):
        self.uuid = uuid
        self.score = score
        self.fact = "a fact whose text must never reach the log"


# ── record shape ──────────────────────────────────────────────────────────────

def test_record_captures_the_query_and_result_uuids():
    r = build_record(tool="recall", query="why BigDecimal?", scopes=["global"],
                     results=[_Result("u1", 0.91), _Result("u2", 0.4)], latency_ms=12.34,
                     project_id="acme-api")
    assert r["tool"] == "recall" and r["query"] == "why BigDecimal?"
    assert r["project_id"] == "acme-api" and r["scopes"] == ["global"]
    assert [h["uuid"] for h in r["top"]] == ["u1", "u2"]
    assert r["top"][0]["score"] == 0.91
    assert r["n_results"] == 2 and r["latency_ms"] == 12.3


def test_record_never_stores_fact_text():
    """The uuids reconstruct anything later analysis needs; prose in a second store is a second
    place for a credential to survive a redaction bug."""
    r = build_record(tool="recall", query="q", scopes=None, results=[_Result("u1", 0.5)],
                     latency_ms=1.0)
    assert "must never reach the log" not in json.dumps(r)


def test_record_redacts_credentials_out_of_the_query_and_keeps_only_the_kind():
    leaked = "why did sk-ant-api03-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA fail?"
    r = build_record(tool="search", query=leaked, scopes=None, results=[], latency_ms=1.0)
    assert "sk-ant-api03-AAAA" not in json.dumps(r)
    assert r.get("redacted_kinds")


def test_clean_queries_carry_no_redaction_marker():
    r = build_record(tool="search", query="how do we handle retries?", scopes=None,
                     results=[], latency_ms=1.0)
    assert "redacted_kinds" not in r


def test_only_the_head_of_the_result_list_is_kept():
    """Storing 50 uuids per read would grow the log for candidates no judge would ever see."""
    results = [_Result(f"u{i}", 0.5) for i in range(40)]
    r = build_record(tool="search", query="q", scopes=None, results=results, latency_ms=1.0)
    assert len(r["top"]) == TOP_N_RESULTS
    assert r["n_results"] == 40, "the true count still gets recorded"


def test_results_without_a_uuid_are_skipped_not_crashed_on():
    r = build_record(tool="search", query="q", scopes=None,
                     results=[_Result("u1", 0.5), object()], latency_ms=1.0)
    assert [h["uuid"] for h in r["top"]] == ["u1"]


def test_a_missing_score_is_recorded_as_null_rather_than_invented():
    r = build_record(tool="search", query="q", scopes=None, results=[_Result("u1")],
                     latency_ms=1.0)
    assert r["top"][0]["score"] is None


# ── write path ────────────────────────────────────────────────────────────────

async def test_record_appends_and_caps_the_list():
    redis = FakeRedis()
    await record(redis, tool="recall", query="q", scopes=None, results=[], latency_ms=1.0)
    assert len(redis.lists[query_log.QUERY_LOG_KEY]) == 1
    assert redis.trims == [(query_log.QUERY_LOG_KEY, 0, MAX_ENTRIES - 1)], \
        "an uncapped log would eventually crowd out the brief cache it shares an instance with"


async def test_record_is_a_no_op_without_redis():
    await record(None, tool="recall", query="q", scopes=None, results=[], latency_ms=1.0)


async def test_a_redis_outage_never_fails_the_read(monkeypatch):
    """Telemetry is not worth failing a retrieval for."""
    monkeypatch.setattr(query_log, "_WARNED", False)
    await record(FakeRedis(fail=True), tool="recall", query="q", scopes=None, results=[],
                 latency_ms=1.0)


async def test_the_outage_warning_is_emitted_once_not_per_read(monkeypatch, caplog):
    monkeypatch.setattr(query_log, "_WARNED", False)
    redis = FakeRedis(fail=True)
    with caplog.at_level("WARNING", logger="synapse.query_log"):
        for _ in range(5):
            await record(redis, tool="recall", query="q", scopes=None, results=[], latency_ms=1.0)
    assert len([r for r in caplog.records if "query log unavailable" in r.message]) == 1


# ── read path ─────────────────────────────────────────────────────────────────

async def test_read_all_returns_records_newest_first():
    redis = FakeRedis()
    for q in ("first", "second"):
        await record(redis, tool="recall", query=q, scopes=None, results=[], latency_ms=1.0)
    assert [e["query"] for e in await read_all(redis)] == ["second", "first"]


async def test_a_malformed_entry_costs_one_sample_not_the_whole_log():
    redis = FakeRedis()
    await record(redis, tool="recall", query="good", scopes=None, results=[], latency_ms=1.0)
    redis.lists[query_log.QUERY_LOG_KEY].append("{not json")
    assert [e["query"] for e in await read_all(redis)] == ["good"]


async def test_read_all_without_redis_is_empty_not_an_error():
    assert await read_all(None) == []


# ── wired into the read path ──────────────────────────────────────────────────

@pytest.fixture
def engine_and_log():
    redis = FakeRedis()
    engine = RetrievalEngine(FakeSearcher([fact("1", "alpha", valid=NOW)]), FakeQueries(),
                             redis=redis)
    return engine, redis


async def test_recall_is_recorded_and_attributed_to_recall(engine_and_log):
    engine, redis = engine_and_log
    await engine.recall("why alpha?", project_id="acme-api", as_of=NOW)
    entries = await read_all(redis)
    assert len(entries) == 1
    assert entries[0]["tool"] == "recall", "recall must not be logged as a bare search"
    assert entries[0]["project_id"] == "acme-api"
    assert entries[0]["query"] == "why alpha?"
    assert entries[0]["top"][0]["uuid"] == "1"


async def test_search_is_recorded_with_its_own_attribution(engine_and_log):
    engine, redis = engine_and_log
    await engine.search("cross-project question", as_of=NOW)
    entries = await read_all(redis)
    assert entries[0]["tool"] == "search" and entries[0]["project_id"] is None


async def test_latency_is_recorded(engine_and_log):
    engine, redis = engine_and_log
    await engine.recall("q", project_id="x", as_of=NOW)
    assert (await read_all(redis))[0]["latency_ms"] >= 0.0


async def test_log_queries_false_records_nothing():
    redis = FakeRedis()
    engine = RetrievalEngine(FakeSearcher([fact("1", "alpha", valid=NOW)]), FakeQueries(),
                             redis=redis, log_queries=False)
    await engine.recall("q", project_id="x", as_of=NOW)
    assert await read_all(redis) == []


# ── the harness must not grade itself on its own homework ─────────────────────

class _EngineWithReader:
    def __init__(self, reader):
        self.reader = reader

    async def recall(self, query, *, project_id=None, limit=10):
        return await self.reader.recall(query, project_id=project_id, limit=limit, as_of=NOW)

    async def search(self, query, *, limit=10):
        return await self.reader.search(query, limit=limit, as_of=NOW)


async def test_run_evaluation_keeps_its_queries_out_of_the_log():
    """Otherwise the held-out set gets mined from the tuned set it exists to be independent of."""
    from synapse.eval.cases import EvalCase
    from synapse.eval.runner import run_evaluation

    redis = FakeRedis()
    reader = RetrievalEngine(FakeSearcher([fact("1", "alpha", valid=NOW)]), FakeQueries(),
                             redis=redis)
    cases = [EvalCase(id="c1", category="acme-api", query="a golden set query",
                      expect_any=["alpha"], project_id="acme-api")]
    await run_evaluation(_EngineWithReader(reader), cases)
    assert await read_all(redis) == []


async def test_run_evaluation_restores_logging_afterwards():
    """The engine outlives the eval; leaving it muted would silently stop real capture."""
    from synapse.eval.runner import run_evaluation

    redis = FakeRedis()
    reader = RetrievalEngine(FakeSearcher([]), FakeQueries(), redis=redis)
    await run_evaluation(_EngineWithReader(reader), [])
    assert reader.log_queries is True

    await reader.recall("a real query", project_id="x", as_of=NOW)
    assert len(await read_all(redis)) == 1


# ── held-out split ────────────────────────────────────────────────────────────

def test_split_is_stable_across_runs_and_insensitive_to_case_and_padding():
    """A split that re-randomises invites rebuilding until it flatters the ranker."""
    from scripts.build_heldout_set import split_of

    assert split_of("Why BigDecimal?") == split_of("  why bigdecimal?  ")
    assert all(split_of("why BigDecimal?") == split_of("why BigDecimal?") for _ in range(5))


def test_split_produces_both_halves_over_realistic_input():
    from scripts.build_heldout_set import split_of

    assignments = {split_of(f"question number {i} about the codebase") for i in range(60)}
    assert assignments == {"dev", "test"}


def test_collect_queries_deduplicates_and_drops_keystrokes():
    from scripts.build_heldout_set import collect_queries

    entries = [
        {"query": "why do we use BigDecimal", "project_id": "acme-api"},
        {"query": "Why do we use BigDecimal", "project_id": "other"},
        {"query": "hi", "project_id": "x"},
        {"query": "", "project_id": "x"},
    ]
    got = collect_queries(entries)
    assert got == [("why do we use BigDecimal", "acme-api")], \
        "case-insensitive dedupe, first project wins, short keystrokes dropped"
