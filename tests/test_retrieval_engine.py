"""Unit tests for the retrieval engine (plan Part 5).

Searcher + graph queries + redis are faked, so ranking, temporal filtering, scope
composition, and brief assembly are tested deterministically. Live ranking over
real Graphiti search lives in scripts/retrieve_smoke.py.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import pytest

from synapse.core.retrieval_engine import (
    Brief,
    CONFIDENCE_WORDS,
    COSINE_BASELINE,
    Fact,
    MEASURED_SAFE_MULTIPLIER,
    Neo4jGraphQueries,
    NodeRow,
    RankWeights,
    RetrievalEngine,
    apply_similarity_floor,
    discriminative_terms,
    lens_ranks,
    mmr_rerank,
    rrf_fuse,
    score_facts,
    temporal_filter,
)
from synapse.core.schema import Scope

NOW = datetime(2026, 5, 31, tzinfo=timezone.utc)


def fact(uuid, text, *, valid=None, invalid=None, created=None, src=None, tgt=None, conf=None,
         score=None, group="project_x", emb=None):
    return Fact(uuid=uuid, fact=text, group_id=group, created_at=created, valid_at=valid,
                invalid_at=invalid, source_uuid=src, target_uuid=tgt, confidence=conf, score=score,
                embedding=emb)


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
    def __init__(self, by_label=None, degrees=None, cross=None, confidence=None):
        self.by_label = by_label or {}
        self._deg = degrees or {}
        self._cross = cross or []
        self._conf = confidence or {}

    async def nodes_by_label(self, labels, scopes, limit):
        # The brief's cross-project section asks for the SHARED tiers only (global, and since
        # Wave 1 the project's cluster) — never a project scope.
        if scopes and all(not s.startswith("project_") for s in scopes):
            return self._cross[:limit]
        return self.by_label.get(labels[0], [])[:limit]

    async def degrees(self, uuids):
        return self._deg

    async def node_confidence(self, uuids):
        return self._conf


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


def test_connectivity_breaks_ties_between_equally_relevant_facts():
    # Connectivity is a MULTIPLICATIVE bonus on the additive base (research §1.2), so it needs a
    # non-zero base to act on. Given equal relevance, the better-connected fact wins.
    w = RankWeights(relevance=1.0, recency=0.0, confidence=0.0, connectivity=0.5)
    facts = [fact("lonely", "x", score=0.9), fact("hub", "y", score=0.9)]
    ranked = score_facts(facts, w, connectivity={"lonely": 0.1, "hub": 1.0}, now=NOW)
    assert ranked[0][0].uuid == "hub"


def test_connectivity_cannot_promote_an_irrelevant_hub_over_a_relevant_fact():
    # The defect this replaced: as an ADDITIVE term, connectivity was a query-independent
    # popularity prior. Measured on the live graph, a fact with relevance 0.027 still scored
    # 0.323 on connectivity alone and outranked far more relevant facts. A bounded multiplier
    # can reorder near-ties but never invert a real relevance gap.
    w = RankWeights()  # shipped defaults
    facts = [
        fact("relevant-loner", "on topic", score=0.95),
        fact("irrelevant-hub", "off topic", score=0.72),
    ]
    ranked = score_facts(facts, w, connectivity={"relevant-loner": 0.0, "irrelevant-hub": 1.0}, now=NOW)
    assert ranked[0][0].uuid == "relevant-loner"


def test_composite_stays_within_zero_and_one_despite_the_bonus():
    w = RankWeights(relevance=1.0, recency=1.0, confidence=1.0, connectivity=1.0)
    facts = [fact("maxed", "x", score=1.0, valid=NOW, conf=1.0)]
    ranked = score_facts(facts, w, connectivity={"maxed": 1.0}, now=NOW)
    assert 0.0 <= ranked[0][1] <= 1.0


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
    # Rescaled from the ABSOLUTE cosine baseline, not min-max normalized (research §1.4):
    # 0.90 -> (0.90-0.65)/0.35 = 0.714, and 0.10 clamps to 0.0. The old min-max put the best
    # hit at exactly 1.0 no matter how mediocre it was, which made scores incomparable
    # across queries and unusable as a threshold.
    assert comp["high"] == pytest.approx((0.90 - COSINE_BASELINE) / (1 - COSINE_BASELINE))
    assert comp["high"] < 1.0
    assert comp["low"] == 0.0


def test_relevance_is_absolute_and_comparable_across_result_sets():
    # A mediocre best-hit must NOT report 1.0 just because it topped a weak set.
    w = RankWeights(relevance=1.0, recency=0.0, confidence=0.0, connectivity=0.0)
    weak = score_facts([fact("a", "x", score=0.73)], w, connectivity={}, now=NOW)
    strong = score_facts([fact("b", "y", score=0.97)], w, connectivity={}, now=NOW)
    assert weak[0][2]["relevance"] < strong[0][2]["relevance"]


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


def test_candidate_multiplier_warns_above_the_measured_safe_width(caplog):
    # Pool width is a SAFETY parameter: 12x was measured to admit cross-project leakage.
    with caplog.at_level(logging.WARNING, logger="synapse.core.retrieval_engine"):
        RetrievalEngine(FakeSearcher([]), FakeQueries(), candidate_multiplier=12)
    assert "measured-safe" in caplog.text


def test_candidate_multiplier_warns_when_set_after_construction(caplog):
    # The A/B harness assigns this attribute directly; an __init__-only guard missed that path
    # entirely, which is why the first version of this warning never fired.
    engine = RetrievalEngine(FakeSearcher([]), FakeQueries())
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="synapse.core.retrieval_engine"):
        engine.candidate_multiplier = 12
    assert "measured-safe" in caplog.text
    assert engine.candidate_multiplier == 12  # a warning, not a veto


def test_candidate_multiplier_silent_at_the_measured_safe_width(caplog):
    with caplog.at_level(logging.WARNING, logger="synapse.core.retrieval_engine"):
        RetrievalEngine(FakeSearcher([]), FakeQueries(),
                        candidate_multiplier=MEASURED_SAFE_MULTIPLIER)
    assert "measured-safe" not in caplog.text


# --- hub-monopoly cap (roadmap item 17) --------------------------------------


def _hub_scored(hub_n=6, other_n=6):
    """Candidates where one source node dominates the high scores."""
    hub = [
        (fact(f"h{i}", f"hub fact {i}", src="HUB", score=0.90 - i * 0.01, created=NOW),
         0.90 - i * 0.01, {})
        for i in range(hub_n)
    ]
    others = [
        (fact(f"o{i}", f"other fact {i}", src=f"SRC{i}", score=0.50 - i * 0.01, created=NOW),
         0.50 - i * 0.01, {})
        for i in range(other_n)
    ]
    return hub + others


def test_source_cap_limits_one_hub_when_the_pool_offers_alternatives():
    got = mmr_rerank(_hub_scored(), 5, lambda_=0.7, max_per_source=2)
    assert [f.source_uuid for f, _s, _c in got].count("HUB") == 2
    assert len(got) == 5  # the slots are filled, not dropped


def test_source_cap_relaxes_rather_than_returning_short():
    # Only two sources exist, so honouring the cap strictly could not fill 5 slots. Returning
    # fewer facts than asked for is the worse failure, so the cap yields.
    scored = _hub_scored(hub_n=6, other_n=1)
    got = mmr_rerank(scored, 5, lambda_=0.7, max_per_source=2)
    assert len(got) == 5
    assert [f.source_uuid for f, _s, _c in got].count("HUB") > 2


def test_source_cap_never_drops_the_top_result():
    got = mmr_rerank(_hub_scored(), 5, lambda_=0.7, max_per_source=1)
    assert got[0][0].uuid == "h0"


def test_source_cap_ignores_facts_with_no_source():
    # An unknown origin is not evidence of a monopoly; capping on it would silently thin results.
    scored = [
        (fact(f"n{i}", f"no source {i}", score=0.9 - i * 0.01, created=NOW), 0.9 - i * 0.01, {})
        for i in range(5)
    ]
    assert len(mmr_rerank(scored, 5, lambda_=0.7, max_per_source=1)) == 5


def test_source_cap_applies_even_with_mmr_disabled():
    # lambda_=1.0 short-circuits MMR, but the cap is a separate concern and must still hold.
    got = mmr_rerank(_hub_scored(), 5, lambda_=1.0, max_per_source=2)
    assert [f.source_uuid for f, _s, _c in got].count("HUB") == 2


# --- rank fusion (roadmap item 17) -------------------------------------------


def test_rrf_fuse_rewards_agreement_across_lenses():
    # b is mid-ranked by every lens; a is first in one and last in the others. Consistent
    # agreement should beat a single lens's enthusiasm.
    fused = rrf_fuse({
        "hybrid": ["a", "b", "c"],
        "cosine": ["c", "b", "a"],
        "lexical": ["c", "b", "a"],
    })
    assert fused["b"] > fused["a"]
    assert fused["c"] == 1.0  # best in the set normalizes to 1.0


def test_rrf_fuse_treats_absence_as_weak_evidence_not_a_veto():
    # `only_hybrid` is missing from two lenses; it still scores, just less.
    fused = rrf_fuse({"hybrid": ["only_hybrid", "both"], "cosine": ["both"]})
    assert 0.0 < fused["only_hybrid"] < fused["both"]


def test_rrf_fuse_empty_and_degenerate_inputs():
    assert rrf_fuse({}) == {}
    assert rrf_fuse({"hybrid": []}) == {}


def test_lens_ranks_lexical_lens_promotes_the_discriminative_match():
    # The live failure: for "what carries events between services?" cosine put an irrelevant
    # short fact above the correct one. The lexical lens is what disagrees.
    facts = _pool(
        fact("junk", "bearer-token is forwarded by service", score=0.7989),
        fact("hit", "Kafka events from LoreVault are consumed by CanonGuard", score=0.7104),
    )
    lenses = lens_ranks(facts, "what carries events between services?")
    assert lenses["cosine"].index("junk") < lenses["cosine"].index("hit")
    assert lenses["lexical"].index("hit") < lenses["lexical"].index("junk")


def test_lens_ranks_hybrid_lens_preserves_searcher_order():
    facts = [fact("first", "a", score=0.1), fact("second", "b", score=0.9)]
    assert lens_ranks(facts, "q")["hybrid"] == ["first", "second"]


def test_score_facts_is_unchanged_when_fusion_carries_no_weight():
    # The mechanism must be inert at fusion=0 so `--weights ...,0.0` is a true control.
    facts = [fact("a", "x", score=0.9, created=NOW), fact("b", "y", score=0.8, created=NOW)]
    w = RankWeights(fusion=0.0)
    plain = score_facts(facts, w, {}, now=NOW)
    with_lenses = score_facts(facts, w, {}, now=NOW, fusion=rrf_fuse(lens_ranks(facts, "x")))
    assert [(f.uuid, s) for f, s, _ in plain] == [(f.uuid, s) for f, s, _ in with_lenses]
    assert "fusion" not in plain[0][2]


def test_score_facts_fusion_can_outvote_a_miscalibrated_cosine():
    # b has the lower cosine but every lens agrees it is the better answer.
    facts = [
        fact("a", "bearer-token is forwarded by service", score=0.80, created=NOW),
        fact("b", "Kafka events are consumed by CanonGuard", score=0.71, created=NOW),
    ]
    fusion = {"a": 0.2, "b": 1.0}
    ranked = score_facts(facts, RankWeights(), {}, now=NOW, fusion=fusion)
    assert ranked[0][0].uuid == "b"
    assert ranked[0][2]["fusion"] == 1.0


# --- rescue band (roadmap item 17) -------------------------------------------


def _pool(*extra):
    """A candidate pool where "service" is common and "ensemble" is rare, as measured live."""
    common = [fact(f"c{i}", f"the service handles request {i}", score=0.90) for i in range(12)]
    return common + list(extra)


def test_discriminative_terms_keeps_rare_and_drops_common():
    pool = _pool(fact("t", "the model uses a soft-voting ensemble", score=0.71))
    terms = discriminative_terms(pool, "which models make up the prediction ensemble?")
    assert "ensemble" in terms          # 1/13 of the pool
    assert "service" not in terms       # 12/13 — present, but discriminates nothing


def test_discriminative_terms_drops_absent_and_question_words():
    pool = _pool()
    terms = discriminative_terms(pool, "what is the best recipe for sourdough bread?")
    # "sourdough"/"bread"/"recipe" appear in nothing (df == 0, they anchor nothing);
    # "best" is a question word. An off-topic query is made entirely of these.
    assert terms == frozenset()


def test_rescue_band_recovers_a_lexically_anchored_below_floor_fact():
    # The live failure this exists for: the correct answer sat at cosine 0.7198 against a
    # 0.72 floor and was deleted two thousandths short.
    target = fact("target", "CalibratedSignalModel uses a LightGBM + XGBoost soft-voting "
                            "ensemble", score=0.7198)
    pool = _pool(target)
    query = "which models make up the prediction ensemble?"
    assert "target" not in {f.uuid for f in apply_similarity_floor(pool, 0.72)}
    kept = {f.uuid for f in apply_similarity_floor(pool, 0.72, query=query, rescue_floor=0.66)}
    assert "target" in kept


def test_rescue_band_does_not_admit_unanchored_junk():
    junk = fact("junk", "Acme-API is built with Spring Boot 3.3", score=0.71)
    pool = _pool(junk)
    kept = {f.uuid for f in apply_similarity_floor(
        pool, 0.72, query="what is the best recipe for sourdough bread?", rescue_floor=0.66)}
    assert "junk" not in kept


def test_rescue_band_respects_its_hard_bottom():
    # Sharing a rare term is evidence, not a licence to reach arbitrarily far down.
    far = fact("far", "an ensemble of unrelated things", score=0.40)
    kept = {f.uuid for f in apply_similarity_floor(
        _pool(far), 0.72, query="prediction ensemble", rescue_floor=0.66)}
    assert "far" not in kept


def test_rescue_band_is_opt_in():
    # Without query/rescue_floor this must behave exactly like the old single threshold, so the
    # A/B control (`run_eval --no-rescue`) measures the real previous behaviour.
    low = fact("low", "a rare ensemble fact", score=0.71)
    assert {f.uuid for f in apply_similarity_floor(_pool(low), 0.72)} == {
        f.uuid for f in _pool() if f.score >= 0.72
    }


def test_relevance_baseline_follows_the_rescue_floor():
    # A rescued fact scores below min_relevance, so rescaling from min_relevance would clamp it to
    # relevance 0.0 and the rescue would buy nothing. The baseline must be the lowest admitted score.
    engine = RetrievalEngine(FakeSearcher([]), FakeQueries(), min_relevance=0.72, rescue_floor=0.66)
    assert engine._relevance_baseline() == 0.66
    engine.rescue_floor = None
    assert engine._relevance_baseline() == 0.72
    engine.min_relevance = 0.0
    assert engine._relevance_baseline() == COSINE_BASELINE


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
        # The "why"/"already rejected" attributes the brief now selects (research §2.1).
        "rationale": None, "alternatives_considered": None, "chosen_over": None,
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


# --- MMR diversity (research §1.3) -------------------------------------------
# The measured defect: recall("monetary values in java") returned FOUR restatements of
# "use BigDecimal for money" in six slots. prompt_recall.py injects only 5 facts per
# prompt, so an agent could spend its whole injected context on one idea.

# Orthogonal-ish 3-d stand-ins for fact embeddings; near-duplicates share a direction.
_MONEY_A = [1.0, 0.0, 0.0]
_MONEY_B = [0.99, 0.01, 0.0]      # a restatement of _MONEY_A
_MONEY_C = [0.98, 0.02, 0.0]      # another restatement
_TESTING = [0.0, 1.0, 0.0]        # a genuinely different topic
_DEPLOY = [0.0, 0.0, 1.0]         # and another


def _scored(*facts):
    """Rank-ordered (fact, composite, components) tuples with descending composites."""
    return [(f, 1.0 - i * 0.01, {}) for i, f in enumerate(facts)]


def test_mmr_replaces_near_duplicates_with_distinct_topics():
    scored = _scored(
        fact("m1", "BigDecimal required for money", emb=_MONEY_A),
        fact("m2", "BigDecimal used in Acme-API for money", emb=_MONEY_B),
        fact("m3", "convention: BigDecimal for monetary values", emb=_MONEY_C),
        fact("t1", "tests run with pytest", emb=_TESTING),
    )
    picked = [f.uuid for f, _s, _c in mmr_rerank(scored, 3, lambda_=0.7)]
    assert picked[0] == "m1", "the top-ranked fact is always kept"
    assert "t1" in picked, "a distinct topic must displace a third restatement"
    assert len([u for u in picked if u.startswith("m")]) < 3


def test_mmr_off_at_lambda_one_preserves_pure_relevance_order():
    scored = _scored(
        fact("m1", "a", emb=_MONEY_A),
        fact("m2", "b", emb=_MONEY_B),
        fact("t1", "c", emb=_TESTING),
    )
    assert [f.uuid for f, _s, _c in mmr_rerank(scored, 3, lambda_=1.0)] == ["m1", "m2", "t1"]


def test_mmr_keeps_all_when_every_fact_is_distinct():
    scored = _scored(
        fact("a", "x", emb=_MONEY_A), fact("b", "y", emb=_TESTING), fact("c", "z", emb=_DEPLOY)
    )
    assert {f.uuid for f, _s, _c in mmr_rerank(scored, 3, lambda_=0.7)} == {"a", "b", "c"}


def test_mmr_degrades_to_relevance_order_without_embeddings():
    # A searcher that could not fetch vectors must not produce arbitrary output.
    scored = _scored(fact("a", "x"), fact("b", "y"), fact("c", "z"))
    assert [f.uuid for f, _s, _c in mmr_rerank(scored, 3, lambda_=0.7)] == ["a", "b", "c"]


def test_mmr_respects_limit_and_handles_empty():
    scored = _scored(fact("a", "x", emb=_MONEY_A), fact("b", "y", emb=_TESTING))
    assert len(mmr_rerank(scored, 1, lambda_=0.7)) == 1
    assert mmr_rerank([], 5) == []
    assert mmr_rerank(scored, 0) == []


async def test_recall_diversifies_over_the_full_candidate_set_not_the_top_k():
    # Diversity must be applied BEFORE truncation, otherwise the duplicates have already
    # consumed the slots by the time we look.
    searcher = FakeSearcher([
        fact("m1", "BigDecimal for money", valid=NOW, score=0.95, emb=_MONEY_A),
        fact("m2", "BigDecimal in Acme-API", valid=NOW, score=0.94, emb=_MONEY_B),
        fact("m3", "BigDecimal convention", valid=NOW, score=0.93, emb=_MONEY_C),
        fact("t1", "pytest for tests", valid=NOW, score=0.80, emb=_TESTING),
    ])
    engine = RetrievalEngine(searcher, FakeQueries(), redis=None, min_relevance=0.0, mmr_lambda=0.7)
    hits = await engine.recall("money", project_id="x", as_of=NOW, limit=2)
    assert len(hits) == 2
    assert "pytest for tests" in [h.fact for h in hits]


# --- confidence resolution (research §1.1) -----------------------------------


def test_confidence_falls_back_to_endpoint_node_confidence():
    # Measured: 0 of 3,011 edges carried confidence, but 165 of 167 Decision NODES did —
    # a real signal the ranker ignored while spending 20% of its weight on the 0.5 default.
    w = RankWeights(relevance=0.0, recency=0.0, confidence=1.0, connectivity=0.0)
    facts = [fact("weak", "x", src="n-tentative"), fact("strong", "y", src="n-locked")]
    node_conf = {"n-tentative": CONFIDENCE_WORDS["tentative"], "n-locked": CONFIDENCE_WORDS["locked"]}
    ranked = score_facts(facts, w, connectivity={}, now=NOW, node_confidence=node_conf)
    assert ranked[0][0].uuid == "strong"
    assert ranked[0][2]["confidence"] == 1.0


def test_edge_confidence_wins_over_node_confidence():
    w = RankWeights(relevance=0.0, recency=0.0, confidence=1.0, connectivity=0.0)
    facts = [fact("f", "x", src="n1", conf=0.9)]
    ranked = score_facts(facts, w, connectivity={}, now=NOW, node_confidence={"n1": 0.4})
    assert ranked[0][2]["confidence"] == 0.9


def test_confidence_defaults_to_neutral_when_nothing_is_known():
    w = RankWeights(relevance=0.0, recency=0.0, confidence=1.0, connectivity=0.0)
    ranked = score_facts([fact("f", "x", src="unknown")], w, connectivity={}, now=NOW)
    assert ranked[0][2]["confidence"] == 0.5


def test_best_endpoint_confidence_is_used():
    w = RankWeights(relevance=0.0, recency=0.0, confidence=1.0, connectivity=0.0)
    facts = [fact("f", "x", src="a", tgt="b")]
    ranked = score_facts(facts, w, connectivity={}, now=NOW,
                         node_confidence={"a": 0.4, "b": 1.0})
    assert ranked[0][2]["confidence"] == 1.0


async def test_recall_survives_a_confidence_lookup_failure():
    class Exploding(FakeQueries):
        async def node_confidence(self, uuids):
            raise RuntimeError("driver down")

    searcher = FakeSearcher([fact("a", "still works", valid=NOW, src="n1")])
    engine = RetrievalEngine(searcher, Exploding(), redis=None)
    hits = await engine.recall("q", project_id="x", as_of=NOW)
    assert [h.fact for h in hits] == ["still works"]


# --- rationale / rejected-alternatives surfacing (research §2.1) --------------


def test_brief_line_surfaces_rationale_and_rejected_alternatives():
    n = NodeRow(
        uuid="d1", name="Graphiti", summary="Use Graphiti for the temporal graph",
        labels=["Decision"], group_id="project_x",
        attributes={"rationale": "temporal model is native",
                    "alternatives_considered": "raw Neo4j, Mem0"},
    )
    line = RetrievalEngine._line(n)
    assert "Use Graphiti for the temporal graph" in line
    assert "why: temporal model is native" in line
    assert "rejected: raw Neo4j, Mem0" in line


def test_brief_line_uses_chosen_over_when_no_alternatives_field():
    n = NodeRow(uuid="t1", name="Neo4j", summary="Neo4j is the graph store",
                labels=["Tool"], group_id="project_x",
                attributes={"chosen_over": "ArangoDB"})
    assert "rejected: ArangoDB" in RetrievalEngine._line(n)


def test_brief_line_unchanged_without_extra_attributes():
    n = NodeRow(uuid="c1", name="conv", summary="Use FastAPI", labels=["Convention"],
                group_id="project_x")
    assert RetrievalEngine._line(n) == "Use FastAPI"


def test_long_rationale_is_trimmed_so_briefs_stay_cheap():
    n = NodeRow(uuid="d2", name="d", summary="s", labels=["Decision"], group_id="project_x",
                attributes={"rationale": "x" * 500})
    line = RetrievalEngine._line(n)
    assert len(line) < 300 and line.endswith("…")


async def test_nodes_by_label_selects_the_why_columns():
    row = {
        "uuid": "d1", "name": "d", "summary": "s", "labels": ["Entity", "Decision"],
        "group_id": "project_x", "created_at": None, "severity": None,
        "rationale": "because latency", "alternatives_considered": "polling",
        "chosen_over": None,
    }
    driver = _FakeDriver([row])
    queries = Neo4jGraphQueries(_FakeGraphiti(driver))
    rows = await queries.nodes_by_label(["Decision"], ["project_x"], 7)
    query_text = driver.queries[0][0]
    assert "n.rationale" in query_text and "n.alternatives_considered" in query_text
    assert rows[0].attributes["rationale"] == "because latency"
    assert rows[0].attributes["alternatives_considered"] == "polling"
    assert "chosen_over" not in rows[0].attributes  # nulls are omitted, not stored as None


async def test_node_confidence_maps_categorical_words_to_floats():
    driver = _FakeDriver([
        {"uuid": "a", "confidence": "settled"},
        {"uuid": "b", "confidence": "LOCKED"},
        {"uuid": "c", "confidence": "nonsense"},
    ])
    queries = Neo4jGraphQueries(_FakeGraphiti(driver))
    out = await queries.node_confidence(["a", "b", "c"])
    assert out == {"a": CONFIDENCE_WORDS["settled"], "b": CONFIDENCE_WORDS["locked"]}


async def test_recall_results_are_presented_in_descending_score_order():
    # MMR chooses WHICH facts survive; it must not leak its marginal-value selection order into
    # the presented order, which surfaced as a non-monotonic score sequence on the live API.
    searcher = FakeSearcher([
        fact("m1", "BigDecimal for money", valid=NOW, score=0.95, emb=_MONEY_A),
        fact("m2", "BigDecimal in Acme-API", valid=NOW, score=0.94, emb=_MONEY_B),
        fact("t1", "pytest for tests", valid=NOW, score=0.80, emb=_TESTING),
        fact("d1", "deploy via docker", valid=NOW, score=0.78, emb=_DEPLOY),
    ])
    engine = RetrievalEngine(searcher, FakeQueries(), redis=None, min_relevance=0.0, mmr_lambda=0.7)
    hits = await engine.recall("q", project_id="x", as_of=NOW, limit=3)
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True), f"not best-first: {scores}"


# --- fact-aware brief (research §2.3 correction, roadmap item 22) --------------
# brief() is 100% label-driven, so knowledge living only on an UNTYPED entity can never reach it —
# 756 of 815 untyped entities carry a real fact. recall() finds those (it searches edge facts);
# only the brief missed them.


def _fq(text, group="project_x"):
    return fact(text[:8], text, group=group, valid=NOW)


class FactQueries(FakeQueries):
    def __init__(self, *args, facts=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._facts = facts or []
        self.recent_calls = []

    async def recent_facts(self, scopes, limit):
        self.recent_calls.append({"scopes": scopes, "limit": limit})
        return list(self._facts)[:limit]


async def test_brief_includes_facts_the_label_sections_cannot_reach():
    q = FactQueries(
        by_label={"Convention": [node("c1", "Use FastAPI", "Convention")]},
        facts=[_fq("Gapping SHORT NQ orders are the designed behavior of the stale-bias watcher")],
    )
    b = await RetrievalEngine(FakeSearcher([]), q, redis=None).brief("x")
    assert any("stale-bias watcher" in line for line in b.recent_facts)


async def test_brief_facts_do_not_repeat_what_is_already_in_the_brief():
    # A fact restating an existing convention line adds nothing but context cost.
    q = FactQueries(
        by_label={"Convention": [node("c1", "BigDecimal is the required type for money in Java",
                                      "Convention")]},
        facts=[
            _fq("BigDecimal is the required type for money in Java"),
            _fq("SSE streaming timeout should be set to 300 seconds"),
        ],
    )
    b = await RetrievalEngine(FakeSearcher([]), q, redis=None).brief("x")
    assert any("SSE streaming" in line for line in b.recent_facts)
    assert not any("BigDecimal" in line for line in b.recent_facts)


async def test_brief_facts_are_internally_diverse():
    q = FactQueries(facts=[
        _fq("TimescaleDB is used for OHLCV bars in the trading stack"),
        _fq("TimescaleDB is used for OHLCV bars across the trading stack"),
        _fq("Redis caches the session brief for thirty minutes"),
    ])
    b = await RetrievalEngine(FakeSearcher([]), q, redis=None).brief("x")
    assert len([line for line in b.recent_facts if "OHLCV" in line]) == 1
    assert any("Redis" in line for line in b.recent_facts)


async def test_brief_facts_are_capped_and_trimmed():
    q = FactQueries(facts=[_fq(f"distinct fact number {i} about subsystem {i}") for i in range(30)]
                    + [_fq("x" * 500)])
    b = await RetrievalEngine(FakeSearcher([]), q, redis=None).brief("x")
    assert len(b.recent_facts) <= 7
    assert all(len(line) <= 201 for line in b.recent_facts)


async def test_brief_composes_the_cluster_tier():
    # Wave 1 wired the cluster tier into recall() but NOT brief() — the one place it matters most.
    q = FactQueries()
    engine = RetrievalEngine(FakeSearcher([]), q, redis=None,
                             cluster_resolver=lambda pid: "trading")
    await engine.brief("acme-api")
    assert q.recent_calls[0]["scopes"] == ["global", "cluster_trading", "project_acme-api"]


async def test_brief_cross_project_section_includes_the_cluster_tier():
    q = FactQueries(cross=[node("p1", "shared pattern", "Pattern")])
    engine = RetrievalEngine(FakeSearcher([]), q, redis=None,
                             cluster_resolver=lambda pid: "trading")
    b = await engine.brief("acme-api")
    # FakeQueries returns `cross` only when asked for the shared scopes; getting a hit proves
    # the shared-scope list was used, and it must now carry the cluster.
    assert b.cross_project_knowledge == ["shared pattern"]


async def test_brief_still_renders_when_the_facts_query_fails():
    class Exploding(FactQueries):
        async def recent_facts(self, scopes, limit):
            raise RuntimeError("driver down")

    q = Exploding(by_label={"Convention": [node("c1", "Use FastAPI", "Convention")]})
    b = await RetrievalEngine(FakeSearcher([]), q, redis=None).brief("x")
    assert b.active_conventions == ["Use FastAPI"] and b.recent_facts == []


async def test_brief_works_against_queries_without_recent_facts_support():
    # An older/partial GraphQueries implementation must not break the killer feature.
    b = await RetrievalEngine(FakeSearcher([]), FakeQueries(), redis=None).brief("x")
    assert b.recent_facts == []


def test_novel_lines_filters_and_limits():
    from synapse.core.retrieval_engine import novel_lines

    out = novel_lines(
        ["the pipeline deduplicates before storing", "redis caches the brief", "unrelated topic"],
        existing=["before storing, the pipeline deduplicates"],
        limit=2,
    )
    assert "redis caches the brief" in out
    assert not any("deduplicates" in line for line in out)
    assert len(out) <= 2


def test_novel_lines_handles_empty_inputs():
    from synapse.core.retrieval_engine import novel_lines

    assert novel_lines([], ["x"], 5) == []
    assert novel_lines(["a fact about something"], [], 0) == []


# The first live run of the facts section filled 6 of 7 slots with one topic. These are the actual
# lines it produced, kept as a regression so the section cannot revert to topic flooding.
_IMPACT_TOPIC = [
    "The square-root market impact law should replace the current linear impact term in Acme-Sim's SlippageModel",
    "The square-root market impact law supersedes the linear impact term (impact_coef * participation)",
    "Acme-Sim's current SlippageModel uses a linear impact term (impact_coef * participation) and should be replaced",
    "The market-impact cost modeling lesson (Bouchaud square-root law, verified June 2026) applies to Acme-Sim",
    "The Bouchaud square-root law (verified June 2026) informed the market-impact cost modeling lesson",
    "The Bouchaud square-root law states that price impact is approximately proportional to sqrt(order size)",
]
_OTHER_TOPICS = [
    "The SEC is a primary regulatory source governing the Reg SHO Rule 203(b)(1) locate requirement.",
    "SSE streaming timeout should be set to 300 seconds",
    "Redis caches the session brief for thirty minutes",
]


def test_brief_facts_cover_topics_instead_of_flooding_one():
    from synapse.core.retrieval_engine import novel_lines

    out = novel_lines(_IMPACT_TOPIC + _OTHER_TOPICS, existing=[], limit=7)
    impact = [line for line in out if "impact" in line.lower() or "bouchaud" in line.lower()]
    assert len(impact) == 1, f"one topic took {len(impact)} slots: {impact}"
    # ...and the genuinely distinct topics all get in.
    assert len(out) >= 4
    for other in _OTHER_TOPICS:
        assert any(other[:24] in line for line in out), other


def test_distinct_topics_sharing_a_word_are_not_over_filtered():
    from synapse.core.retrieval_engine import novel_lines

    out = novel_lines(
        [
            "Redis caches the session brief for thirty minutes",
            "Redis is also the Celery broker for background curation tasks",
        ],
        existing=[], limit=5,
    )
    assert len(out) == 2, "sharing one salient token must not collapse two distinct facts"


async def test_live_shaped_brief_does_not_flood_one_topic():
    q = FactQueries(facts=[_fq(t) for t in _IMPACT_TOPIC + _OTHER_TOPICS])
    b = await RetrievalEngine(FakeSearcher([]), q, redis=None).brief("x")
    impact = [line for line in b.recent_facts if "bouchaud" in line.lower() or "impact" in line.lower()]
    assert len(impact) == 1
