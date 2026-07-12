"""Unit tests for the retrieval engine (plan Part 5).

Searcher + graph queries + redis are faked, so ranking, temporal filtering, scope
composition, and brief assembly are tested deterministically. Live ranking over
real Graphiti search lives in scripts/retrieve_smoke.py.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from synapse.core.schema import Scope
from synapse.core.retrieval_engine import (
    Brief,
    Fact,
    Neo4jGraphQueries,
    NodeRow,
    RankWeights,
    RetrievalEngine,
    apply_similarity_floor,
    score_facts,
    temporal_filter,
)

NOW = datetime(2026, 5, 31, tzinfo=timezone.utc)


def fact(uuid, text, *, valid=None, invalid=None, created=None, src=None, tgt=None, conf=None,
         score=None, group="project_x"):
    return Fact(uuid=uuid, fact=text, group_id=group, created_at=created, valid_at=valid,
                invalid_at=invalid, source_uuid=src, target_uuid=tgt, confidence=conf, score=score)


def node(name, summary, label, *, severity=None, group="project_x"):
    attrs = {"severity": severity} if severity else {}
    return NodeRow(uuid=name, name=name, summary=summary, labels=[label], group_id=group, attributes=attrs)


# --- fakes -------------------------------------------------------------------


class FakeSearcher:
    def __init__(self, facts):
        self._facts = facts
        self.last = None
        self.calls = []

    async def search(self, query, scopes, limit, center_node_uuid):
        self.last = {"query": query, "scopes": scopes, "limit": limit, "center": center_node_uuid}
        self.calls.append(dict(self.last))
        # Emulate a DB fetch cap: return at most `limit` facts (like num_results).
        return list(self._facts)[:limit]


class FakeQueries:
    def __init__(self, by_label=None, degrees=None, cross=None):
        self.by_label = by_label or {}
        self._deg = degrees or {}
        self._cross = cross or []

    async def nodes_by_label(self, labels, scopes, limit):
        if scopes == [Scope.GLOBAL]:
            return self._cross[:limit]
        return self.by_label.get(labels[0], [])[:limit]

    async def degrees(self, uuids):
        return self._deg


class FakeRedis:
    def __init__(self):
        self.store = {}

    async def get(self, k):
        return self.store.get(k)

    async def set(self, k, v, ex=None):
        self.store[k] = v

    async def delete(self, k):
        self.store.pop(k, None)


# --- temporal filter ---------------------------------------------------------


def test_temporal_filter_point_in_time():
    facts = [
        fact("a", "current", valid=NOW - timedelta(days=10)),                    # valid now
        fact("b", "future", valid=NOW + timedelta(days=5)),                       # not yet valid
        fact("c", "superseded", valid=NOW - timedelta(days=40), invalid=NOW - timedelta(days=5)),  # expired
        fact("d", "untimed"),                                                     # no bounds → always
    ]
    kept = {f.uuid for f in temporal_filter(facts, NOW)}
    assert kept == {"a", "d"}


def test_temporal_filter_as_of_history():
    # As of 20 days ago, the superseded fact 'c' was still valid.
    as_of = NOW - timedelta(days=20)
    facts = [fact("c", "superseded", valid=NOW - timedelta(days=40), invalid=NOW - timedelta(days=5))]
    assert [f.uuid for f in temporal_filter(facts, as_of)] == ["c"]


# --- ranking -----------------------------------------------------------------


def test_score_facts_recency_breaks_ties():
    w = RankWeights(relevance=0.0, recency=1.0, confidence=0.0, connectivity=0.0)
    facts = [fact("old", "x", valid=NOW - timedelta(days=365)), fact("new", "y", valid=NOW)]
    ranked = score_facts(facts, w, connectivity={}, now=NOW)
    assert ranked[0][0].uuid == "new"


def test_score_facts_connectivity_weighting():
    w = RankWeights(relevance=0.0, recency=0.0, confidence=0.0, connectivity=1.0)
    facts = [fact("lonely", "x"), fact("hub", "y")]
    ranked = score_facts(facts, w, connectivity={"lonely": 0.1, "hub": 1.0}, now=NOW)
    assert ranked[0][0].uuid == "hub"


# --- recall (semantic + graph + scope) --------------------------------------


async def test_recall_returns_relevant_ranked():
    searcher = FakeSearcher([fact("1", "alpha", valid=NOW), fact("2", "beta", valid=NOW)])
    engine = RetrievalEngine(searcher, FakeQueries(), redis=None)
    hits = await engine.recall("anything", project_id="x", as_of=NOW, limit=10)
    assert [h.fact for h in hits] == ["alpha", "beta"]
    assert all(0.0 <= h.score <= 1.0 for h in hits)


async def test_recall_composes_scopes():
    searcher = FakeSearcher([])
    engine = RetrievalEngine(searcher, FakeQueries(), redis=None)
    await engine.recall("q", project_id="acme-store", agent_role="planner", as_of=NOW)
    assert searcher.last["scopes"] == ["global", "project_acme-store", "agent_planner"]


async def test_recall_passes_center_node_for_graph_traversal():
    searcher = FakeSearcher([fact("1", "connected", valid=NOW)])
    engine = RetrievalEngine(searcher, FakeQueries(degrees={"n": 3}), redis=None)
    await engine.recall("q", project_id="x", center_node_uuid="node-123", as_of=NOW)
    assert searcher.last["center"] == "node-123"


async def test_recall_temporal_excludes_future_and_expired():
    searcher = FakeSearcher([
        fact("a", "current", valid=NOW - timedelta(days=1)),
        fact("b", "expired", valid=NOW - timedelta(days=9), invalid=NOW - timedelta(days=2)),
    ])
    engine = RetrievalEngine(searcher, FakeQueries(), redis=None)
    hits = await engine.recall("q", project_id="x", as_of=NOW)
    assert [h.fact for h in hits] == ["current"]


# --- scope composition -------------------------------------------------------


def test_scope_compose():
    assert Scope.compose("acme-store") == ["global", "project_acme-store"]
    assert Scope.compose() == ["global"]
    assert Scope.compose("gf", "planner") == ["global", "project_gf", "agent_planner"]


# --- brief -------------------------------------------------------------------


def _brief_queries():
    return FakeQueries(
        by_label={
            "Convention": [node("conv", "Always use FastAPI", "Convention")],
            "Decision": [node("dec", "Use SQLite over Postgres", "Decision")],
            "Lesson": [
                node("l_low", "minor note", "Lesson", severity="low"),
                node("l_crit", "AdMob test mode or ban", "Lesson", severity="critical"),
            ],
        },
        cross=[node("pat", "Keystore backup in 3 places", "Pattern", group="global")],
    )


async def test_brief_structured_and_useful():
    engine = RetrievalEngine(FakeSearcher([]), _brief_queries(), redis=None)
    brief = await engine.brief("acme-store")

    assert isinstance(brief, Brief)
    assert brief.active_conventions == ["Always use FastAPI"]
    assert brief.key_decisions == ["Use SQLite over Postgres"]
    # lessons sorted by severity → critical first
    assert brief.relevant_lessons[0] == "AdMob test mode or ban"
    assert brief.cross_project_knowledge == ["Keystore backup in 3 places"]
    assert "acme-store" in brief.project_summary


async def test_brief_uses_and_busts_redis_cache():
    redis = FakeRedis()
    engine = RetrievalEngine(FakeSearcher([]), _brief_queries(), redis=redis)

    first = await engine.brief("acme-store")
    assert first.cached is False
    assert "brief:acme-store" in redis.store

    second = await engine.brief("acme-store")
    assert second.cached is True  # served from cache

    await engine.invalidate_brief("acme-store")
    assert "brief:acme-store" not in redis.store


# --- WP-A: real relevance scores ---------------------------------------------


def test_relevance_uses_real_scores():
    # Positional order is [low, high] but the real similarity scores invert it:
    # 'high' has the larger score, so relevance must rank it first.
    w = RankWeights(relevance=1.0, recency=0.0, confidence=0.0, connectivity=0.0)
    facts = [fact("low", "x", score=0.10), fact("high", "y", score=0.90)]
    ranked = score_facts(facts, w, connectivity={}, now=NOW)
    assert ranked[0][0].uuid == "high"
    comp = {f.uuid: c["relevance"] for f, _, c in ranked}
    # min-max normalized: hi -> 1.0, lo -> 0.0
    assert comp["high"] == 1.0
    assert comp["low"] == 0.0


def test_relevance_falls_back_to_positional():
    # No real scores → fall back to positional 1 - i/n (incoming order wins).
    w = RankWeights(relevance=1.0, recency=0.0, confidence=0.0, connectivity=0.0)
    facts = [fact("first", "x"), fact("second", "y"), fact("third", "z")]
    ranked = score_facts(facts, w, connectivity={}, now=NOW)
    assert [f.uuid for f, _, _ in ranked] == ["first", "second", "third"]
    comp = {f.uuid: c["relevance"] for f, _, c in ranked}
    assert comp["first"] == 1.0  # 1 - 0/3
    assert comp["second"] > comp["third"]


def test_relevance_missing_score_gets_half():
    # Mixed: facts with a score are min-max normalized; a None score gets 0.5.
    w = RankWeights(relevance=1.0, recency=0.0, confidence=0.0, connectivity=0.0)
    facts = [fact("a", "x", score=0.0), fact("b", "y", score=1.0), fact("c", "z", score=None)]
    ranked = score_facts(facts, w, connectivity={}, now=NOW)
    comp = {f.uuid: c["relevance"] for f, _, c in ranked}
    assert comp["a"] == 0.0
    assert comp["b"] == 1.0
    assert comp["c"] == 0.5


# --- WP-A: back-fill after filtering -----------------------------------------


async def test_recall_backfills_when_top_hits_invalidated():
    # limit=3, candidate_multiplier=1 → initial fetch cap = 3. The first 3 facts
    # (which saturate that cap) are all expired, yielding 0 valid; the surviving
    # valid fact sits at index 3, past the cap. A back-fill must re-fetch with a
    # doubled multiplier (cap = 6) and surface 'keep'.
    facts = [
        fact(f"exp{i}", f"expired{i}", valid=NOW - timedelta(days=9),
             invalid=NOW - timedelta(days=2))
        for i in range(3)
    ]
    facts.append(fact("keep", "surviving", valid=NOW - timedelta(days=1)))
    facts += [
        fact(f"pad{i}", f"expired-pad{i}", valid=NOW - timedelta(days=9),
             invalid=NOW - timedelta(days=2))
        for i in range(5)
    ]
    searcher = FakeSearcher(facts)
    engine = RetrievalEngine(searcher, FakeQueries(), redis=None, candidate_multiplier=1)
    hits = await engine.recall("q", project_id="x", as_of=NOW, limit=3)

    assert [h.fact for h in hits] == ["surviving"]
    assert len(searcher.calls) == 2  # one back-fill round
    assert searcher.calls[1]["limit"] == 6  # multiplier doubled: 3 * (1*2)


async def test_recall_no_backfill_when_enough_survive():
    # All facts valid → no back-fill, single fetch.
    facts = [fact(str(i), f"f{i}", valid=NOW) for i in range(5)]
    searcher = FakeSearcher(facts)
    engine = RetrievalEngine(searcher, FakeQueries(), redis=None, candidate_multiplier=3)
    hits = await engine.recall("q", project_id="x", as_of=NOW, limit=3)
    assert len(hits) == 3
    assert len(searcher.calls) == 1


# --- similarity floor ---------------------------------------------------------


def test_similarity_floor_drops_below_threshold():
    # Off-topic junk (cosine ~0.68) is dropped; real hits (~0.85) survive.
    facts = [
        fact("junk1", "off-topic a", score=0.68),
        fact("junk2", "off-topic b", score=0.71),
        fact("hit1", "real a", score=0.85),
        fact("hit2", "real b", score=0.80),
    ]
    kept = {f.uuid for f in apply_similarity_floor(facts, 0.72)}
    assert kept == {"hit1", "hit2"}


def test_similarity_floor_disabled_when_zero_or_none():
    facts = [fact("a", "x", score=0.1), fact("b", "y", score=0.9)]
    assert len(apply_similarity_floor(facts, 0.0)) == 2
    assert len(apply_similarity_floor(facts, None)) == 2


def test_similarity_floor_keeps_facts_without_a_score():
    # No absolute similarity available → never had a signal to reject on → keep it,
    # so a searcher that can't compute cosine degrades to no-floor, not empty.
    facts = [fact("scored_low", "x", score=0.10), fact("unscored", "y", score=None)]
    kept = {f.uuid for f in apply_similarity_floor(facts, 0.72)}
    assert kept == {"unscored"}


async def test_search_applies_floor_to_off_topic_junk():
    # A full result set of low-similarity junk returns EMPTY once the floor engages,
    # instead of confident junk.
    searcher = FakeSearcher([
        fact("j1", "junk", valid=NOW, score=0.66),
        fact("j2", "junk", valid=NOW, score=0.69),
    ])
    engine = RetrievalEngine(searcher, FakeQueries(), redis=None, min_relevance=0.72)
    hits = await engine.recall("off-topic", project_id="x", as_of=NOW)
    assert hits == []


async def test_search_floor_preserves_real_hits():
    searcher = FakeSearcher([
        fact("good", "real", valid=NOW, score=0.85),
        fact("junk", "noise", valid=NOW, score=0.60),
    ])
    engine = RetrievalEngine(searcher, FakeQueries(), redis=None, min_relevance=0.72)
    hits = await engine.recall("q", project_id="x", as_of=NOW)
    assert [h.fact for h in hits] == ["real"]


async def test_search_floor_off_by_default_when_no_scores():
    # Facts with no scores are untouched by the floor (positional-relevance fallback).
    searcher = FakeSearcher([fact("a", "alpha", valid=NOW), fact("b", "beta", valid=NOW)])
    engine = RetrievalEngine(searcher, FakeQueries(), redis=None, min_relevance=0.72)
    hits = await engine.recall("q", project_id="x", as_of=NOW)
    assert [h.fact for h in hits] == ["alpha", "beta"]


# --- WP-A: connectivity default ----------------------------------------------


def test_connectivity_default_zero():
    # A fact absent from the connectivity map gets 0.0 (no free median boost).
    w = RankWeights(relevance=0.0, recency=0.0, confidence=0.0, connectivity=1.0)
    facts = [fact("known", "x"), fact("unknown", "y")]
    ranked = score_facts(facts, w, connectivity={"known": 1.0}, now=NOW)
    comp = {f.uuid: c["connectivity"] for f, _, c in ranked}
    assert comp["known"] == 1.0
    assert comp["unknown"] == 0.0


# --- WP-A: brief validity filter (real Cypher via a fake driver) -------------


class _FakeRecords:
    def __init__(self, records):
        self.records = records


class _FakeDriver:
    """Records the last executed query; returns pre-canned rows."""

    def __init__(self, rows):
        self._rows = rows
        self.queries = []

    async def execute_query(self, query, **params):
        self.queries.append((query, params))
        return _FakeRecords(self._rows)


class _FakeGraphiti:
    def __init__(self, driver):
        self.driver = driver


async def test_brief_excludes_invalidated_and_archived():
    # The real nodes_by_label query must filter out invalidated + archived nodes
    # and must NOT ship the embedding-laden properties(n) blob over the wire.
    row = {
        "uuid": "n1", "name": "conv", "summary": "Use FastAPI",
        "labels": ["Entity", "Convention"], "group_id": "project_x", "created_at": None,
        "severity": None,  # Neo4j returns the key with a null value for missing props
    }
    driver = _FakeDriver([row])
    queries = Neo4jGraphQueries(_FakeGraphiti(driver))
    rows = await queries.nodes_by_label(["Convention"], ["project_x"], 7)

    query_text = driver.queries[0][0]
    assert "n.invalid_at IS NULL" in query_text
    assert "coalesce(n.archived, false) = false" in query_text
    assert "properties(n)" not in query_text  # no embedding blob shipped
    assert rows[0].uuid == "n1" and rows[0].labels == ["Convention"]
    assert rows[0].summary == "Use FastAPI"
