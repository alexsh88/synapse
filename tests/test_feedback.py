"""Retrieval feedback (roadmap item 14, research §6).

The point of these tests is mostly about what must NOT happen: the signal must not become
self-referential (eval/UI reads inflating the counters of the facts they measure), and a feedback
write must never be able to fail a read.
"""

from __future__ import annotations

from datetime import datetime, timezone

from synapse.core.feedback import FactFeedback, FeedbackSummary
from synapse.core.retrieval_engine import Fact, RetrievalEngine

NOW = datetime(2026, 7, 25, tzinfo=timezone.utc)


class _Searcher:
    def __init__(self, facts):
        self._facts = facts

    async def search(self, query, scopes, limit, center_node_uuid):
        return list(self._facts)[:limit]


class _Queries:
    def __init__(self, explode=False):
        self.recorded: list[list[str]] = []
        self.explode = explode

    async def nodes_by_label(self, labels, scopes, limit):
        return []

    async def degrees(self, uuids):
        return {}

    async def node_confidence(self, uuids):
        return {}

    async def recent_facts(self, scopes, limit):
        return []

    async def record_recall(self, fact_uuids):
        if self.explode:
            raise RuntimeError("neo4j down")
        self.recorded.append(list(fact_uuids))


def _facts():
    return [
        Fact(uuid="f1", fact="alpha", group_id="project_x", valid_at=NOW, score=0.9),
        Fact(uuid="f2", fact="beta", group_id="project_x", valid_at=NOW, score=0.88),
    ]


async def test_impressions_are_not_recorded_by_default():
    # THE key invariant: eval runs and UI browsing must not inflate the counters of the very facts
    # they measure, or the signal becomes self-referential within a day.
    q = _Queries()
    engine = RetrievalEngine(_Searcher(_facts()), q, redis=None, min_relevance=0.0)
    await engine.recall("q", project_id="x", as_of=NOW)
    assert q.recorded == []


async def test_impressions_are_recorded_when_a_real_consumer_asks():
    q = _Queries()
    engine = RetrievalEngine(_Searcher(_facts()), q, redis=None, min_relevance=0.0)
    hits = await engine.recall("q", project_id="x", as_of=NOW, feedback=True)
    assert q.recorded == [[h.uuid for h in hits]]


async def test_only_the_served_facts_count_not_the_whole_candidate_set():
    # Impressions must reflect what the consumer actually saw, post-truncation.
    q = _Queries()
    engine = RetrievalEngine(_Searcher(_facts()), q, redis=None, min_relevance=0.0)
    await engine.recall("q", project_id="x", as_of=NOW, limit=1, feedback=True)
    assert len(q.recorded[0]) == 1


async def test_an_empty_result_records_nothing():
    q = _Queries()
    engine = RetrievalEngine(_Searcher([]), q, redis=None)
    await engine.recall("q", project_id="x", as_of=NOW, feedback=True)
    assert q.recorded == []


async def test_a_feedback_write_failure_never_fails_the_read():
    # Retrieval is the core value (R7); feedback is bookkeeping.
    q = _Queries(explode=True)
    engine = RetrievalEngine(_Searcher(_facts()), q, redis=None, min_relevance=0.0)
    hits = await engine.recall("q", project_id="x", as_of=NOW, feedback=True)
    assert [h.fact for h in hits] == ["alpha", "beta"]


async def test_queries_without_record_recall_support_are_tolerated():
    class Old:
        async def nodes_by_label(self, labels, scopes, limit):
            return []

        async def degrees(self, uuids):
            return {}

    engine = RetrievalEngine(_Searcher(_facts()), Old(), redis=None, min_relevance=0.0)
    hits = await engine.recall("q", project_id="x", as_of=NOW, feedback=True)
    assert len(hits) == 2


# --- the derived judgements ---------------------------------------------------


def test_a_corrected_fact_is_suspect_however_popular():
    assert FactFeedback(uuid="f", recalled_n=99, corrected_n=1).is_suspect
    assert not FactFeedback(uuid="f", recalled_n=99, corrected_n=0).is_suspect


def test_a_never_served_fact_is_dead_weight():
    assert FactFeedback(uuid="f", recalled_n=0).is_dead_weight
    assert not FactFeedback(uuid="f", recalled_n=1).is_dead_weight


def test_coverage_is_the_headline_number():
    # A large corpus with low coverage is mostly write-only.
    assert FeedbackSummary(total_facts=200, ever_recalled=50).coverage == 0.25
    assert FeedbackSummary().coverage == 0.0  # no division by zero on an empty graph


async def test_the_recall_impression_query_is_one_batched_write():
    from synapse.core.retrieval_engine import Neo4jGraphQueries

    class _Driver:
        def __init__(self):
            self.calls = []

        async def execute_query(self, query, **params):
            self.calls.append((query, params))

            class _R:
                records = []

            return _R()

    class _G:
        def __init__(self, d):
            self.driver = d

    driver = _Driver()
    await Neo4jGraphQueries(_G(driver)).record_recall(["a", "b", "c"])
    assert len(driver.calls) == 1, "a recall must cost one extra write, not one per fact"
    query, params = driver.calls[0]
    assert "UNWIND" in query and params["uuids"] == ["a", "b", "c"]
    assert "recalled_n" in query and "last_recalled_at" in query


async def test_no_write_at_all_for_an_empty_uuid_list():
    from synapse.core.retrieval_engine import Neo4jGraphQueries

    class _Driver:
        calls: list = []

        async def execute_query(self, query, **params):
            self.calls.append(query)

    class _G:
        driver = _Driver()

    await Neo4jGraphQueries(_G()).record_recall([])
    assert _G.driver.calls == []


async def test_forgetting_a_fact_counts_as_a_correction():
    # An explicit forget is a judgement that the fact was wrong — the strongest signal available.
    from synapse.core.knowledge_engine import KnowledgeEngine

    class _Driver:
        def __init__(self):
            self.queries = []

        async def execute_query(self, query, **params):
            self.queries.append(query)

            class _R:
                records = [{"uuid": "f1"}]

            return _R()

    class _G:
        def __init__(self, d):
            self.driver = d

    driver = _Driver()
    engine = KnowledgeEngine.__new__(KnowledgeEngine)
    engine.graphiti = _G(driver)
    out = await KnowledgeEngine.forget(engine, "f1", reason="wrong")
    assert out["success"]
    assert any("corrected_n" in q and "last_corrected_at" in q for q in driver.queries)


async def test_corrections_are_counted_over_all_edges_not_just_active_ones():
    # Caught live: a correction (update/forget) DEACTIVATES the fact it corrects, so counting
    # corrections over active edges only made the figure structurally ~0 — the summary reported
    # corrected_facts=0 while the suspect list showed a corrected fact.
    from synapse.core.graph_queries import GraphService

    class _Driver:
        def __init__(self):
            self.queries = []

        async def execute_query(self, query, **params):
            self.queries.append(query)
            active = "e.invalid_at IS NULL" in query
            corrected_count = "RETURN count(e) AS corrected" in query

            class _R:
                records = (
                    [{"total": 100, "ever": 10, "impressions": 42}] if active and not corrected_count
                    else [{"corrected": 7}] if corrected_count
                    else []
                )

            return _R()

    class _G:
        def __init__(self, d):
            self.driver = d

    driver = _Driver()
    summary = await GraphService(_G(driver)).feedback()
    assert summary.corrected_facts == 7
    assert summary.total_facts == 100 and summary.total_impressions == 42

    corrections_query = next(q for q in driver.queries if "RETURN count(e) AS corrected" in q)
    assert "e.invalid_at IS NULL" not in corrections_query, (
        "the corrections count must not be restricted to active edges"
    )
