"""Unit tests for the eval scoring logic — pure, no engine.

Covers:
- Original hit@k / MRR behaviour (regression guard on existing tests)
- keyword-set (AND) matching semantics
- must_not_match violation counting
- precision@k calculation
- Baseline load / compare_to_baseline regression-gate logic
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from synapse.eval.cases import EvalCase
from synapse.eval.runner import (
    CI_MIN_N,
    BaselineHasViolations,
    CaseOutcome,
    EvalReport,
    _hit_rate_of,
    _mrr_of,
    bootstrap_ci,
    compare_to_baseline,
    evaluate_case,
    load_baseline,
    save_baseline,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class R:
    """Minimal result stub with .fact and .scope attributes."""
    def __init__(self, fact: str, scope: str = "project_x"):
        self.fact = fact
        self.scope = scope


def _case(**kw) -> EvalCase:
    base: dict = dict(id="t", category="acme-api", query="q", expect_any=["x"])
    base.update(kw)
    return EvalCase(**base)


# ---------------------------------------------------------------------------
# Original hit@k / MRR tests (must keep passing)
# ---------------------------------------------------------------------------

def test_hit_at_rank_1():
    out = evaluate_case(_case(expect_any=["BigDecimal"]), [R("Use BigDecimal for money"), R("other")])
    assert out.hit and out.rank == 1


def test_hit_at_rank_3_case_insensitive():
    out = evaluate_case(_case(expect_any=["hexagonal"]),
                        [R("a"), R("b"), R("Services use HEXAGONAL architecture")])
    assert out.hit and out.rank == 3


def test_miss_returns_no_rank():
    out = evaluate_case(_case(expect_any=["zzz"]), [R("a"), R("b")])
    assert not out.hit and out.rank is None and "no expected substring" in out.note


def test_scope_filter_rejects_wrong_scope():
    c = _case(category="cross_project", expect_any=["timescale"], expect_scope="project_acme-api")
    # right content, wrong scope -> miss
    assert not evaluate_case(c, [R("TimescaleDB gotcha", scope="project_acme-data")]).hit
    # right content in the right scope (rank 2) -> hit
    out = evaluate_case(c, [R("x"), R("TimescaleDB gotcha", scope="project_acme-api")])
    assert out.hit and out.rank == 2


def test_metrics_aggregate_and_mrr():
    outs = [
        evaluate_case(_case(id="a", expect_any=["x"]), [R("x")]),             # hit @1
        evaluate_case(_case(id="b", expect_any=["zz"]), [R("a")]),             # miss
        evaluate_case(_case(id="c", category="global", expect_any=["q"]),
                      [R("a"), R("q here")]),                                   # hit @2
    ]
    m = EvalReport(outcomes=outs).metrics()
    assert m["acme-api"]["n"] == 2 and m["acme-api"]["hits"] == 1 and m["acme-api"]["hit_rate"] == 0.5
    assert m["global"]["mrr"] == 0.5            # single hit at rank 2 -> 1/2
    assert m["OVERALL"]["n"] == 3 and m["OVERALL"]["hits"] == 2
    # MRR overall = (1/1 + 0 + 1/2) / 3 = 0.5
    assert m["OVERALL"]["mrr"] == 0.5


# ---------------------------------------------------------------------------
# keyword-set (AND) matching
# ---------------------------------------------------------------------------

def test_keywords_all_present_is_hit():
    """ALL keywords must appear for a keyword-set hit."""
    c = _case(expect_any=[], keywords=["fair value", "gap"])
    out = evaluate_case(c, [R("ICT Fair Value Gap is a key concept"), R("other")])
    assert out.hit and out.rank == 1


def test_keywords_partial_match_is_miss():
    """Only one of the required keywords present → NOT a hit."""
    c = _case(expect_any=[], keywords=["fair value", "order block"])
    out = evaluate_case(c, [R("fair value gap explained")])
    assert not out.hit


def test_keywords_case_insensitive():
    c = _case(expect_any=[], keywords=["ORDER BLOCK", "liquidity"])
    out = evaluate_case(c, [R("Order block with liquidity sweep")])
    assert out.hit and out.rank == 1


def test_keywords_fallback_when_expect_any_fails():
    """keywords match succeeds even when expect_any would fail."""
    c = _case(expect_any=["nonexistent"], keywords=["hexagonal", "adapter"])
    out = evaluate_case(c, [R("Hexagonal architecture uses adapter pattern")])
    assert out.hit and out.rank == 1


def test_expect_any_wins_even_when_keywords_fail():
    """expect_any match succeeds even when keywords would fail."""
    c = _case(expect_any=["BigDecimal"], keywords=["missing_kw"])
    out = evaluate_case(c, [R("Use BigDecimal for money")])
    assert out.hit and out.rank == 1


def test_no_positive_spec_returns_hit_false():
    """A case with no expect_any and no keywords never registers a hit."""
    c = _case(expect_any=[], keywords=[])
    out = evaluate_case(c, [R("anything at all")])
    assert not out.hit
    assert out.precision_at_k is None


# ---------------------------------------------------------------------------
# must_not_match violation counting
# ---------------------------------------------------------------------------

def test_no_violations_when_clean():
    c = _case(expect_any=["x"], must_not_match=["forbidden"])
    out = evaluate_case(c, [R("x is fine"), R("also fine")])
    assert out.violations == 0


def test_single_violation_detected():
    c = _case(expect_any=["x"], must_not_match=["forbidden"])
    out = evaluate_case(c, [R("x is here"), R("this is forbidden content")])
    assert out.violations == 1


def test_multiple_violations_counted_per_result():
    """Each forbidden substring in each result counts as a separate violation."""
    c = _case(expect_any=["x"], must_not_match=["unity", "monobehaviour"])
    results = [
        R("Unity scene with MonoBehaviour"),   # 2 violations
        R("Unity prefab"),                      # 1 violation
        R("clean result x"),                    # 0 violations
    ]
    out = evaluate_case(c, results)
    assert out.violations == 3


def test_violations_case_insensitive():
    c = _case(expect_any=["x"], must_not_match=["workflow"])
    out = evaluate_case(c, [R("x Java WORKFLOW ENGINE")])
    assert out.violations == 1


def test_violation_with_no_hit_still_counted():
    """Violations are counted even when the case has no positive hit."""
    c = _case(expect_any=[], keywords=[], must_not_match=["java", "spring"])
    out = evaluate_case(c, [R("Java Spring Boot app")])
    assert out.violations == 2
    assert not out.hit


def test_violations_aggregate_in_metrics():
    """EvalReport.metrics() sums violations across the group."""
    c1 = _case(id="a", expect_any=["x"], must_not_match=["bad"])
    c2 = _case(id="b", expect_any=["y"], must_not_match=["evil"])
    outs = [
        evaluate_case(c1, [R("x"), R("bad result")]),   # 1 violation
        evaluate_case(c2, [R("y"), R("evil thing")]),   # 1 violation
    ]
    m = EvalReport(outcomes=outs).metrics()
    assert m["OVERALL"]["violations"] == 2
    assert m["acme-api"]["violations"] == 2


# ---------------------------------------------------------------------------
# precision@k
# ---------------------------------------------------------------------------

def test_precision_at_k_all_relevant():
    """All results match → precision = 1.0."""
    c = _case(expect_any=["x"])
    out = evaluate_case(c, [R("x here"), R("also x"), R("x again")])
    assert out.precision_at_k == 1.0


def test_precision_at_k_half_relevant():
    c = _case(expect_any=["x"])
    out = evaluate_case(c, [R("x here"), R("irrelevant"), R("x again"), R("noise")])
    # 2 out of 4 match
    assert out.precision_at_k == 0.5


def test_precision_at_k_none_for_negative_cases():
    """Negative-only cases (no expect_any, no keywords) get precision=None."""
    c = _case(expect_any=[], keywords=[], must_not_match=["bad"])
    out = evaluate_case(c, [R("anything")])
    assert out.precision_at_k is None


def test_precision_at_k_aggregated_in_metrics():
    """EvalReport.metrics() averages precision@k over cases with positive specs."""
    c1 = _case(id="a", expect_any=["x"])
    c2 = _case(id="b", expect_any=["y"])
    c3 = _case(id="c", expect_any=[], keywords=[])   # no positive spec
    outs = [
        evaluate_case(c1, [R("x"), R("irrelevant")]),   # precision = 0.5
        evaluate_case(c2, [R("y"), R("y again")]),       # precision = 1.0
        evaluate_case(c3, [R("something")]),             # precision = None (excluded)
    ]
    m = EvalReport(outcomes=outs).metrics()
    # avg of [0.5, 1.0] = 0.75
    assert m["OVERALL"]["precision_at_k"] == pytest.approx(0.75, abs=1e-4)


def test_precision_at_k_none_when_all_negative():
    c = _case(expect_any=[], keywords=[])
    outs = [evaluate_case(c, [R("a"), R("b")])]
    m = EvalReport(outcomes=outs).metrics()
    assert m["OVERALL"]["precision_at_k"] is None


# ---------------------------------------------------------------------------
# Baseline: save / load / compare
# ---------------------------------------------------------------------------

def _make_metrics(hit_rate: float, mrr: float, violations: int = 0) -> dict:
    return {
        "OVERALL": {
            "n": 10,
            "hits": int(hit_rate * 10),
            "hit_rate": hit_rate,
            "mrr": mrr,
            "precision_at_k": None,
            "violations": violations,
        }
    }


def test_save_and_load_baseline(tmp_path):
    path = tmp_path / "baseline.json"
    metrics = _make_metrics(0.8, 0.7)
    save_baseline(metrics, path)
    loaded = load_baseline(path)
    assert loaded == metrics


def test_load_baseline_returns_none_when_missing(tmp_path):
    assert load_baseline(tmp_path / "nonexistent.json") is None


def test_no_regression_when_metrics_equal():
    m = _make_metrics(0.8, 0.7)
    assert compare_to_baseline(m, m) == []


def test_no_regression_within_threshold():
    current = _make_metrics(0.77, 0.68)    # 3.75% and 2.9% drops — within 5%
    baseline = _make_metrics(0.80, 0.70)
    assert compare_to_baseline(current, baseline) == []


def test_regression_on_hit_rate_drop():
    current = _make_metrics(0.70, 0.70)    # 12.5% relative drop in hit_rate
    baseline = _make_metrics(0.80, 0.70)
    regressions = compare_to_baseline(current, baseline)
    assert len(regressions) == 1
    assert "hit_rate" in regressions[0]


def test_regression_on_mrr_drop():
    current = _make_metrics(0.80, 0.60)    # 14.3% relative drop in MRR
    baseline = _make_metrics(0.80, 0.70)
    regressions = compare_to_baseline(current, baseline)
    assert len(regressions) == 1
    assert "mrr" in regressions[0]


def test_regression_when_both_drop():
    current = _make_metrics(0.70, 0.60)
    baseline = _make_metrics(0.80, 0.70)
    regressions = compare_to_baseline(current, baseline)
    assert len(regressions) == 2


def test_regression_on_violations_increase():
    current = _make_metrics(0.80, 0.70, violations=3)
    baseline = _make_metrics(0.80, 0.70, violations=0)
    regressions = compare_to_baseline(current, baseline)
    assert len(regressions) == 1
    assert "violations" in regressions[0]


# The two tests below asserted the OPPOSITE until 2026-07-25: violations were compared relative
# to the baseline, so a recorded leak became permission for that many leaks forever. Scope
# isolation (R5) is a correctness property, not a metric — see compare_to_baseline's docstring.


def test_regression_when_violations_merely_persist():
    current = _make_metrics(0.80, 0.70, violations=2)
    baseline = _make_metrics(0.80, 0.70, violations=2)
    assert compare_to_baseline(current, baseline) != []


def test_regression_even_when_violations_decrease():
    # Fewer leaks is progress, not success. Two projects can still see each other's knowledge.
    current = _make_metrics(0.80, 0.70, violations=1)
    baseline = _make_metrics(0.80, 0.70, violations=3)
    assert compare_to_baseline(current, baseline) != []


def test_no_regression_when_violations_are_zero():
    clean = _make_metrics(0.80, 0.70, violations=0)
    assert compare_to_baseline(clean, clean) == []


def test_save_baseline_refuses_to_enshrine_violations():
    # The hole this closes: the gate only compared against the recorded number, so one
    # --save-baseline on a leaking config would silence it permanently.
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "baseline.json"
        with pytest.raises(BaselineHasViolations):
            save_baseline(_make_metrics(0.80, 0.70, violations=1), path)
        assert not path.exists()
        save_baseline(_make_metrics(0.80, 0.70, violations=0), path)
        assert json.loads(path.read_text())["OVERALL"]["violations"] == 0


def test_regression_threshold_exactly_at_boundary():
    """Values well within the 5% relative threshold should NOT trigger."""
    # 3% relative drop of 0.80: 0.80 * 0.97 = 0.776 — clearly within threshold
    current = _make_metrics(0.776, 0.70)
    baseline = _make_metrics(0.80, 0.70)
    # drop = (0.80 - 0.776) / 0.80 = 0.03 — not > 0.05 → no regression
    assert compare_to_baseline(current, baseline) == []


def test_regression_just_over_threshold():
    """Just beyond 5% threshold should trigger."""
    current = _make_metrics(0.759, 0.70)   # (0.80 - 0.759) / 0.80 ≈ 0.05125 > 0.05
    baseline = _make_metrics(0.80, 0.70)
    regressions = compare_to_baseline(current, baseline)
    assert any("hit_rate" in r for r in regressions)


def test_compare_handles_missing_overall_key():
    """If baseline has no OVERALL key, compare returns no regressions."""
    current = _make_metrics(0.5, 0.5)
    assert compare_to_baseline(current, {}) == []


def test_compare_handles_zero_baseline_metric():
    """Zero baseline: any drop is a regression; same or better is not."""
    baseline = _make_metrics(0.0, 0.0)
    same = _make_metrics(0.0, 0.0)
    better = _make_metrics(0.5, 0.5)
    assert compare_to_baseline(same, baseline) == []
    assert compare_to_baseline(better, baseline) == []


# ---------------------------------------------------------------------------
# Negative case integration smoke-test (no engine — just verify the cases load)
# ---------------------------------------------------------------------------

def test_negative_cases_in_golden_set():
    from synapse.eval.cases import GOLDEN_SET
    neg = [c for c in GOLDEN_SET if c.category == "negative"]
    assert len(neg) >= 5, "Need at least 5 negative cases"
    # Every negative case must have must_not_match
    for c in neg:
        assert c.must_not_match, f"Negative case {c.id} has no must_not_match"


def test_neg_cross_project_leakage_case_violations():
    """Verify a cross-project leakage case correctly flags out-of-scope terms."""
    from synapse.eval.cases import GOLDEN_SET
    # Accept either the private (acme-flow/acme-store) or demo (api/infra) leakage cases.
    leakage_ids = ("neg-acme-flow-leakage", "neg-api-infra-leakage")
    case = next((c for c in GOLDEN_SET if c.id in leakage_ids), None)
    assert case is not None, f"No leakage case found; expected one of {leakage_ids}"
    # Build a result that matches at least one must_not_match token from the case.
    bad_token = case.must_not_match[0]
    results = [R(f"This result mentions {bad_token} which should not appear")]
    out = evaluate_case(case, results)
    assert out.violations >= 1
    assert not out.hit   # no positive spec


def test_neg_reverse_leakage_case_violations():
    """Verify the reverse cross-project leakage case correctly flags out-of-scope terms."""
    from synapse.eval.cases import GOLDEN_SET
    leakage_ids = ("neg-acme-store-leakage", "neg-infra-web-leakage")
    case = next((c for c in GOLDEN_SET if c.id in leakage_ids), None)
    assert case is not None, f"No reverse leakage case found; expected one of {leakage_ids}"
    bad_token = case.must_not_match[0]
    results = [R(f"Result about {bad_token} which should not surface here")]
    out = evaluate_case(case, results)
    assert out.violations >= 1
    assert not out.hit


def test_neg_off_topic_produces_no_violations_on_clean_results():
    """Off-topic case returns 0 violations if the engine correctly returns nothing."""
    from synapse.eval.cases import GOLDEN_SET
    # Accept either the private (cooking/medical) or demo (cooking/gardening) off-topic cases.
    off_topic_ids = ("neg-off-topic-cooking", "neg-off-topic-gardening")
    case = next((c for c in GOLDEN_SET if c.id in off_topic_ids), None)
    assert case is not None, f"No off-topic case found; expected one of {off_topic_ids}"
    out = evaluate_case(case, [R("no knowledge found")])
    assert out.violations == 0
    assert not out.hit


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------

def _outcomes(hits: int, misses: int, *, rank: int = 1) -> list[CaseOutcome]:
    return (
        [CaseOutcome(id=f"h{i}", category="acme-api", hit=True, rank=rank) for i in range(hits)]
        + [CaseOutcome(id=f"m{i}", category="acme-api", hit=False) for i in range(misses)]
    )


def test_ci_is_none_below_the_minimum_sample_size():
    """An interval over a handful of cases spans nearly everything; None says so honestly."""
    assert bootstrap_ci(_outcomes(CI_MIN_N - 1, 0), _hit_rate_of) is None


def test_ci_is_reported_at_the_minimum_sample_size():
    assert bootstrap_ci(_outcomes(CI_MIN_N, 0), _hit_rate_of) is not None


def test_ci_is_deterministic_across_runs():
    """A jittering interval cannot be diffed against the baseline, which is the whole use."""
    group = _outcomes(30, 20)
    assert bootstrap_ci(group, _hit_rate_of) == bootstrap_ci(group, _hit_rate_of)


def test_ci_brackets_the_point_estimate():
    group = _outcomes(30, 20)
    lo, hi = bootstrap_ci(group, _hit_rate_of)
    assert lo <= _hit_rate_of(group) <= hi


def test_ci_collapses_when_every_case_agrees():
    """No sampling error to estimate when every resample is identical."""
    assert bootstrap_ci(_outcomes(40, 0), _hit_rate_of) == (1.0, 1.0)
    assert bootstrap_ci(_outcomes(0, 40), _hit_rate_of) == (0.0, 0.0)


def test_ci_narrows_as_the_sample_grows():
    """The property that makes it worth reporting: more cases, less uncertainty."""
    # Same resample count on both sides so the only difference is the sample size, and fewer
    # than the default because this is the one test whose cost scales with both.
    small_lo, small_hi = bootstrap_ci(_outcomes(6, 6), _hit_rate_of, resamples=2000)
    large_lo, large_hi = bootstrap_ci(_outcomes(150, 150), _hit_rate_of, resamples=2000)
    assert (large_hi - large_lo) < (small_hi - small_lo)


def test_mrr_ci_reflects_rank_quality():
    """Rank-3 hits must produce a strictly lower interval than rank-1 hits."""
    top = bootstrap_ci(_outcomes(20, 0, rank=1), _mrr_of)
    deep = bootstrap_ci(_outcomes(20, 0, rank=3), _mrr_of)
    assert deep[1] < top[0]


def test_metrics_carry_intervals_for_overall_but_not_tiny_categories():
    report = EvalReport(outcomes=_outcomes(20, 10) + [
        CaseOutcome(id="solo", category="synapse", hit=True, rank=1)
    ])
    m = report.metrics()
    assert m["OVERALL"]["hit_rate_ci"] is not None
    assert m["OVERALL"]["mrr_ci"] is not None
    assert m["synapse"]["hit_rate_ci"] is None, "n=1 must not get an interval"


def test_format_prints_the_interval():
    report = EvalReport(outcomes=_outcomes(20, 10))
    assert "95% CI" in report.format()


def test_baseline_round_trips_with_intervals(tmp_path):
    """Intervals are tuples in Python and lists in JSON — the gate must survive the round trip."""
    path = tmp_path / "baseline.json"
    metrics = EvalReport(outcomes=_outcomes(20, 10)).metrics()
    save_baseline(metrics, path)
    loaded = load_baseline(path)
    assert loaded["OVERALL"]["hit_rate_ci"] == list(metrics["OVERALL"]["hit_rate_ci"])
    assert compare_to_baseline(metrics, loaded) == []
