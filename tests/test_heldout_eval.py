"""Pooling and the two-judge protocol — the machinery behind a held-out eval set.

No LLM calls: the judges are injected, which is the point of taking them as a dict.
"""

from __future__ import annotations

from synapse.eval.judge import (
    ABSTAIN,
    Judgement,
    adversarial_probe,
    agreement_report,
    cohens_kappa,
    judge_one,
    parse_grade,
)
from synapse.eval.pooling import Strategy, pool_all, pool_query, pool_stats


class Hit:
    def __init__(self, uuid, fact="a fact", scope="global"):
        self.uuid = uuid
        self.fact = fact
        self.scope = scope


def strat(name, *result_sets):
    """A strategy returning a fixed list, ignoring the query."""
    hits = [Hit(u) for u in result_sets]

    async def run(query, project_id):
        return hits

    return Strategy(name=name, run=run)


# ── pooling ───────────────────────────────────────────────────────────────────

async def test_pool_is_the_union_of_every_strategy():
    pool = await pool_query("q", None, [strat("a", "1", "2"), strat("b", "2", "3")])
    assert {c.uuid for c in pool.candidates} == {"1", "2", "3"}


async def test_pool_records_which_strategies_found_each_candidate_and_at_what_rank():
    pool = await pool_query("q", None, [strat("a", "1", "2"), strat("b", "2", "1")])
    by_uuid = {c.uuid: c for c in pool.candidates}
    assert by_uuid["1"].found_by == {"a": 1, "b": 2}
    assert by_uuid["2"].found_by == {"a": 2, "b": 1}


async def test_corroborated_candidates_are_presented_before_one_off_finds():
    """A human spot-check of the top of the pool should hit the strongest candidates first."""
    pool = await pool_query("q", None, [strat("a", "1", "9"), strat("b", "1", "8")])
    assert pool.candidates[0].uuid == "1"
    assert pool.candidates[0].strategy_count == 2


async def test_depth_bounds_each_strategy_contribution():
    wide = strat("wide", *[str(i) for i in range(30)])
    pool = await pool_query("q", None, [wide], depth=5)
    assert len(pool.candidates) == 5


async def test_a_failing_strategy_does_not_shrink_the_pool_silently(caplog):
    async def boom(query, project_id):
        raise RuntimeError("searcher down")

    with caplog.at_level("WARNING", logger="synapse.eval.pooling"):
        pool = await pool_query("q", None, [strat("ok", "1"), Strategy("broken", boom)])
    assert {c.uuid for c in pool.candidates} == {"1"}
    assert any("broken" in r.getMessage() for r in caplog.records)


async def test_results_without_a_uuid_are_skipped():
    async def run(query, project_id):
        return [object(), Hit("1")]

    pool = await pool_query("q", None, [Strategy("odd", run)])
    assert [c.uuid for c in pool.candidates] == ["1"]


async def test_pool_all_handles_many_queries():
    pools = await pool_all([("q1", "p"), ("q2", None)], [strat("a", "1")])
    assert [p.query for p in pools] == ["q1", "q2"]
    assert pools[0].project_id == "p" and pools[1].project_id is None


async def test_pool_stats_expose_whether_pooling_was_worth_doing():
    """unique_share near zero means the strategies are one ranker wearing hats."""
    identical = await pool_all([("q", None)], [strat("a", "1", "2"), strat("b", "1", "2")])
    assert pool_stats(identical)["unique_to_one_strategy"] == 0

    diverse = await pool_all([("q", None)], [strat("a", "1"), strat("b", "2")])
    stats = pool_stats(diverse)
    assert stats["unique_to_one_strategy"] == 2
    assert stats["unique_share"] == 1.0
    assert stats["candidates_per_query_mean"] == 2.0


# ── grade parsing ─────────────────────────────────────────────────────────────

def test_parse_grade_reads_a_bare_digit():
    assert parse_grade("2") == 2


def test_parse_grade_survives_the_prose_models_add_anyway():
    assert parse_grade("Grade: 1 — related but not an answer.") == 1


def test_parse_grade_abstains_rather_than_guessing():
    assert parse_grade("") == ABSTAIN
    assert parse_grade("I cannot grade this") == ABSTAIN


# ── judging ───────────────────────────────────────────────────────────────────

def fake_judge(grade):
    async def run(query, fact):
        return grade

    return run


def broken_judge():
    async def run(query, fact):
        raise RuntimeError("model unavailable")

    return run


async def test_judge_one_collects_a_grade_from_every_judge():
    j = await judge_one("q", "u1", "f", {"a": fake_judge(2), "b": fake_judge(2)})
    assert j.grades == {"a": 2, "b": 2} and j.agreed


async def test_a_failed_judge_abstains_instead_of_falling_back():
    """Falling back to the other judge would collapse two families into one and hide it."""
    j = await judge_one("q", "u1", "f", {"a": fake_judge(2), "b": broken_judge()})
    assert j.grades["b"] == ABSTAIN
    assert not j.agreed, "one grade plus an abstention is not agreement"


def test_consensus_takes_the_lower_grade_on_disagreement():
    j = Judgement(uuid="u", query="q", grades={"a": 2, "b": 1})
    assert j.consensus == 1, "under-counting relevance makes the resulting score a floor"


def test_consensus_ignores_abstentions():
    assert Judgement("u", "q", {"a": 2, "b": ABSTAIN}).consensus == 2


# ── agreement statistics ──────────────────────────────────────────────────────

def test_kappa_is_one_for_perfect_agreement_across_varied_grades():
    assert cohens_kappa([0, 1, 2, 0, 1, 2], [0, 1, 2, 0, 1, 2]) == 1.0


def test_kappa_is_negative_when_graders_systematically_disagree():
    assert cohens_kappa([0, 1, 0, 1], [1, 0, 1, 0]) == -1.0


def test_kappa_does_not_reward_two_graders_who_always_say_the_same_thing():
    """Raw agreement is 100% here; kappa's whole job is to notice there is no signal."""
    assert cohens_kappa([0, 0, 0, 0], [0, 0, 0, 0]) == 1.0
    assert cohens_kappa([0, 0, 0, 0], [0, 0, 0, 1]) == 0.0


def test_kappa_drops_pairs_where_either_judge_abstained():
    assert cohens_kappa([1, 1, ABSTAIN], [1, 1, 0]) == cohens_kappa([1, 1], [1, 1])


def test_kappa_is_none_when_too_little_survives():
    assert cohens_kappa([ABSTAIN, 1], [0, ABSTAIN]) is None


def test_agreement_report_carries_everything_a_reader_needs():
    js = [
        Judgement("u1", "q", {"claude": 2, "gemma": 2}),
        Judgement("u2", "q", {"claude": 0, "gemma": 1}),
        Judgement("u3", "q", {"claude": 1, "gemma": ABSTAIN}),
    ]
    r = agreement_report(js, "claude", "gemma")
    assert r["n"] == 3 and r["n_gradeable"] == 2 and r["abstentions"] == 1
    assert r["exact_agreement"] == 0.5
    assert "cohens_kappa" in r
    assert r["grade_distribution"]["gemma"] == {2: 1, 1: 1}


# ── adversarial probe ─────────────────────────────────────────────────────────

async def test_probe_reports_the_share_of_wrong_facts_a_judge_accepts():
    """LoCoMo's judge accepted 62.8% of deliberately wrong answers; this is how you find out."""
    pairs = [("q1", "decoy"), ("q2", "decoy"), ("q3", "decoy"), ("q4", "decoy")]
    report = await adversarial_probe(pairs, {"lenient": fake_judge(2), "strict": fake_judge(0)})
    assert report["probes"] == 4
    assert report["per_judge"]["lenient"]["acceptance_rate"] == 1.0
    assert report["per_judge"]["strict"]["acceptance_rate"] == 0.0


async def test_probe_counts_partial_relevance_as_acceptance():
    """Grade 1 on a fact that answers nothing is still the judge being fooled."""
    report = await adversarial_probe([("q", "decoy")], {"soft": fake_judge(1)})
    assert report["per_judge"]["soft"]["acceptance_rate"] == 1.0


async def test_probe_excludes_abstentions_from_the_rate():
    report = await adversarial_probe([("q", "d"), ("q2", "d")],
                                     {"flaky": fake_judge(ABSTAIN)})
    assert report["per_judge"]["flaky"]["n"] == 0
    assert report["per_judge"]["flaky"]["acceptance_rate"] is None
