"""Eval runner — run the golden set against the engine and score retrieval quality.

Metrics
-------
hit@k
    Fraction of cases where an expected fact appears in the top-k results.
    A result is a hit when:
    * any substring in ``expect_any`` appears in the fact (case-insensitive), OR
    * ALL strings in ``keywords`` appear in the fact (case-insensitive AND-match).
    Scope filtering (``expect_scope``) is applied after a content match: a result
    with the right content but the wrong scope is skipped and the search continues.

MRR
    Mean reciprocal rank of the first expected hit across all cases.
    Rewards ranking the right answer higher.

precision@k
    Fraction of the top-k results that are "relevant" (match any expected
    substring or keyword set) for that case.  Computed only for positive cases
    (those with at least one entry in ``expect_any`` or ``keywords``).

diversity@k
    Distinct ideas in the top-k, divided by k.  Near-duplicate facts are collapsed by
    token-set Jaccard, so four restatements of one fact score 0.25 rather than 1.0.
    This is the counterweight to precision@k, which *punishes* diversity: displacing
    redundant matches with new information reads as a precision loss.  Added 2026-07-25
    with the MMR pass (research §1.3/§6) — without it the harness could only see MMR's
    cost, never its benefit.

violations
    Count of ``must_not_match`` substrings (case-insensitive) found across
    ALL top-k results for a case.  Each individual occurrence in each result
    counts separately.  A perfect run has 0 violations.

Baseline / regression gate
--------------------------
Running against a baseline (``synapse/eval/baseline.json``) compares the current
run's metrics to the stored ones and exits non-zero if quality regressed:

* hit@k or MRR drops > 5 % *relative* (e.g. 0.80 → below 0.76), OR
* total violations increase compared to the baseline.

Workflow
--------
Run against the live engine::

    python -m scripts.run_eval

Save current metrics as the new baseline::

    python -m scripts.run_eval --save-baseline

Thresholds (hard-coded, not per-case)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* hit@k regression: more than 5 % relative drop (``REGRESSION_THRESHOLD = 0.05``)
* MRR regression:   more than 5 % relative drop
* violations:       any increase vs. baseline
"""

from __future__ import annotations

import json
import math
import random
import re
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from synapse.eval.cases import EvalCase

BASELINE_PATH = Path(__file__).parent / "baseline.json"
REGRESSION_THRESHOLD = 0.05   # 5 % relative drop triggers gate failure

# ── Bootstrap confidence intervals ────────────────────────────────────────────
# A point estimate over ~50 cases invites a precision it has not earned: "MRR 0.743" reads as
# though the third digit means something. The percentile bootstrap resamples the case set with
# replacement and reports the interval the metric actually occupies, which is the difference
# between "0.743" and "0.743, and a rerun on a different 52 cases could plausibly land anywhere
# in [0.63, 0.85]". It costs nothing to compute and it is what stops the gate's numbers from
# being quoted as a quality score.
BOOTSTRAP_RESAMPLES = 10_000
# Fixed so the same report is byte-identical across runs — a CI that jitters per run cannot be
# diffed, and the whole point is comparing this run to the baseline.
BOOTSTRAP_SEED = 20260819
# Below this, the interval spans nearly the whole range and communicates only "we don't know".
# Reporting None is more honest than reporting [0.0, 1.0] as though it were a measurement.
CI_MIN_N = 10


class CaseOutcome(BaseModel):
    id: str
    category: str
    hit: bool
    rank: int | None = None          # 1-based rank of first expected hit, else None
    top_fact: str = ""
    note: str = ""
    precision_at_k: float | None = None   # None when no positive expectation
    violations: int = 0                   # count of must_not_match hits in top-k
    diversity_at_k: float | None = None   # distinct ideas / results (None when no results)


# Token-set Jaccard at or above this counts two facts as restatements of one idea.
# Lexical rather than embedding-based on purpose: the metric must be explainable and must not
# depend on the same embedding space whose behaviour it is measuring.
_IDEA_OVERLAP = 0.6
# Function words carry no topical signal, so they must not inflate the overlap between two
# otherwise-unrelated facts.
_STOPWORDS = frozenset(
    "a an the is are was were be been being for to of in on at by with from as and or not "
    "this that these those it its must should can will do does did has have had than then "
    "when while all any into over under via per use used uses using".split()
)


def _idea_tokens(fact: str) -> frozenset[str]:
    words = re.findall(r"[a-z0-9]+", fact.lower())
    return frozenset(w for w in words if w not in _STOPWORDS and len(w) > 2)


def _distinct_ideas(facts: list[str]) -> int:
    """Count how many genuinely distinct ideas a result list contains.

    Greedy single-link clustering: a fact joins an existing cluster when its token-set Jaccard
    against any member reaches :data:`_IDEA_OVERLAP`.
    """
    clusters: list[list[frozenset[str]]] = []
    for fact in facts:
        tokens = _idea_tokens(fact)
        if not tokens:
            continue
        for cluster in clusters:
            for member in cluster:
                union = tokens | member
                if union and len(tokens & member) / len(union) >= _IDEA_OVERLAP:
                    cluster.append(tokens)
                    break
            else:
                continue
            break
        else:
            clusters.append([tokens])
    return len(clusters)


def _result_matches(fact: str, case: EvalCase) -> bool:
    """Return True when *fact* satisfies the case's positive matching criteria."""
    fact_lower = fact.lower()
    if case.expect_any and any(n.lower() in fact_lower for n in case.expect_any):
        return True
    if case.keywords and all(kw.lower() in fact_lower for kw in case.keywords):
        return True
    return False


def evaluate_case(case: EvalCase, results: list) -> CaseOutcome:
    """Score one case against an ordered result list (objects with .fact/.scope).

    Handles:
    * ``expect_any`` substring matching (original behaviour, preserved)
    * ``keywords``   AND-set matching
    * ``expect_scope`` scope filtering
    * ``must_not_match`` violation counting across all top-k results
    * ``precision@k`` calculation
    """
    # --- Count violations across all top-k results ---
    violations = 0
    if case.must_not_match:
        for r in results:
            fact_lower = (getattr(r, "fact", "") or "").lower()
            for forbidden in case.must_not_match:
                if forbidden.lower() in fact_lower:
                    violations += 1

    # --- Determine whether positive matching is expected at all ---
    has_positive_spec = bool(case.expect_any or case.keywords)

    # --- Scan for a hit ---
    hit = False
    hit_rank: int | None = None
    top_fact = getattr(results[0], "fact", "") if results else ""
    note = ""
    hit_count = 0   # for precision@k

    for i, r in enumerate(results):
        fact = (getattr(r, "fact", "") or "")
        scope = getattr(r, "scope", "") or ""
        if has_positive_spec and _result_matches(fact, case):
            hit_count += 1
            if not hit:
                # First hit — record rank but only accept if scope matches
                if case.expect_scope and case.expect_scope not in scope:
                    pass   # right content, wrong scope — keep looking for first accepted hit
                else:
                    hit = True
                    hit_rank = i + 1

    if has_positive_spec and not hit:
        note = "no expected substring in top-k"
        if case.expect_scope:
            note += f" from scope {case.expect_scope}"

    # precision@k: fraction of top-k that matched (only meaningful for positive cases)
    precision = None
    if has_positive_spec and results:
        precision = round(hit_count / len(results), 4)

    # diversity@k: distinct ideas / results. precision@k alone PUNISHES diversity — when several
    # near-duplicate facts all match the expected substring, displacing them looks like a loss
    # even though the result set became more informative. This is the counterweight, and it
    # applies to every case (including negatives, where crowding is still waste).
    diversity = None
    if results:
        facts = [(getattr(r, "fact", "") or "") for r in results]
        diversity = round(_distinct_ideas(facts) / len(results), 4)

    return CaseOutcome(
        id=case.id,
        category=case.category,
        hit=hit,
        rank=hit_rank,
        top_fact=top_fact,
        note=note,
        precision_at_k=precision,
        violations=violations,
        diversity_at_k=diversity,
    )


def _hit_rate_of(group: Sequence[CaseOutcome]) -> float:
    return sum(o.hit for o in group) / len(group) if group else 0.0


def _mrr_of(group: Sequence[CaseOutcome]) -> float:
    if not group:
        return 0.0
    return sum(1.0 / o.rank for o in group if o.hit and o.rank) / len(group)


def bootstrap_ci(
    group: Sequence[CaseOutcome],
    statistic: Callable[[Sequence[CaseOutcome]], float],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    alpha: float = 0.05,
) -> tuple[float, float] | None:
    """Percentile bootstrap CI for *statistic* over *group*, or None below ``CI_MIN_N``.

    Resamples the CASES with replacement, not the results within a case: the case set is the
    thing a rerun would vary, so it is the thing whose sampling error we are estimating. Two
    systems whose intervals overlap are not distinguishable at this sample size, which is the
    claim this function exists to let the report make.
    """
    n = len(group)
    if n < CI_MIN_N:
        return None
    rng = random.Random(seed)
    stats = sorted(
        statistic([group[rng.randrange(n)] for _ in range(n)]) for _ in range(resamples)
    )
    lo_i = int((alpha / 2) * resamples)
    hi_i = min(int((1 - alpha / 2) * resamples), resamples - 1)
    return (round(stats[lo_i], 3), round(stats[hi_i], 3))


class EvalReport(BaseModel):
    outcomes: list[CaseOutcome] = Field(default_factory=list)

    def metrics(self) -> dict[str, dict[str, Any]]:
        cats = sorted({o.category for o in self.outcomes})
        out: dict[str, dict] = {}
        for cat in [*cats, "OVERALL"]:
            group = self.outcomes if cat == "OVERALL" else [o for o in self.outcomes if o.category == cat]
            n = len(group)
            hits = sum(o.hit for o in group)
            mrr = sum((1.0 / o.rank) for o in group if o.hit and o.rank) / n if n else 0.0
            # precision@k: average over cases that have a positive spec
            # Bind the narrowed floats rather than re-reading the Optional attribute inside
            # sum(): the comprehension proves it is not None, but only for the value it yields.
            precs = [o.precision_at_k for o in group if o.precision_at_k is not None]
            avg_prec = (sum(precs) / len(precs)) if precs else None
            total_violations = sum(o.violations for o in group)
            divs = [o.diversity_at_k for o in group if o.diversity_at_k is not None]
            avg_div = (sum(divs) / len(divs)) if divs else None
            out[cat] = {
                "n": n,
                "hits": hits,
                "hit_rate": round(hits / n, 3) if n else 0.0,
                "mrr": round(mrr, 3),
                "precision_at_k": round(avg_prec, 4) if avg_prec is not None else None,
                "violations": total_violations,
                "diversity_at_k": round(avg_div, 4) if avg_div is not None else None,
                # None below CI_MIN_N — most per-project categories hold a handful of cases,
                # and an interval over 3 of them would be decoration, not evidence.
                "hit_rate_ci": bootstrap_ci(group, _hit_rate_of),
                "mrr_ci": bootstrap_ci(group, _mrr_of),
            }
        return out

    def format(self) -> str:
        m = self.metrics()
        lines = ["Retrieval quality (golden set):", ""]
        lines.append(
            f"  {'category':<16} {'n':>3} {'hits':>5} {'hit@k':>7} {'MRR':>6}"
            f" {'prec@k':>8} {'div@k':>7} {'violations':>11}"
        )
        for cat, v in m.items():
            prec = f"{v['precision_at_k']:.4f}" if v["precision_at_k"] is not None else "   n/a"
            div = f"{v['diversity_at_k']:.3f}" if v.get("diversity_at_k") is not None else "  n/a"
            lines.append(
                f"  {cat:<16} {v['n']:>3} {v['hits']:>5} {v['hit_rate']:>7}"
                f" {v['mrr']:>6} {prec:>8} {div:>7} {v['violations']:>11}"
            )
        overall = m.get("OVERALL", {})
        hr_ci, mrr_ci = overall.get("hit_rate_ci"), overall.get("mrr_ci")
        if hr_ci and mrr_ci:
            lines.append(
                f"\n  95% CI (percentile bootstrap, {BOOTSTRAP_RESAMPLES:,} resamples, "
                f"n={overall.get('n')}):  hit@k [{hr_ci[0]}, {hr_ci[1]}]"
                f"   MRR [{mrr_ci[0]}, {mrr_ci[1]}]"
            )
        misses = [o for o in self.outcomes if not o.hit and (o.note or o.violations)]
        if misses:
            lines.append("\n  misses / violations:")
            for o in misses:
                suffix = f" [violations={o.violations}]" if o.violations else ""
                lines.append(f"    x {o.id} ({o.category}) - {o.note}{suffix}")
        viol_cases = [o for o in self.outcomes if o.violations]
        for o in viol_cases:
            if o.hit:   # show violations even when the positive hit passes
                lines.append(f"    ! {o.id} ({o.category}) - {o.violations} violation(s)")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Baseline support
# ---------------------------------------------------------------------------

def load_baseline(path: Path = BASELINE_PATH) -> dict | None:
    """Return the stored baseline metrics dict, or None if no file exists."""
    if not path.exists():
        return None
    with path.open() as fh:
        return json.load(fh)


class BaselineHasViolations(ValueError):
    """Raised when a baseline save would enshrine a scope-isolation violation."""


def save_baseline(metrics: dict[str, dict], path: Path = BASELINE_PATH) -> None:
    """Persist *metrics* to *path* as the new baseline.

    Refuses to save metrics containing violations. This closes the hole that made
    :func:`compare_to_baseline`'s violation rule bypassable: the gate only ever compared against
    the recorded number, so one ``--save-baseline`` run on a leaking config would have made that
    leak the new normal and silenced the gate permanently. A violation is a correctness failure,
    not a metric to bank — fix the leak, then save.
    """
    violations = (metrics.get("OVERALL", {}) or {}).get("violations", 0) or 0
    if violations:
        raise BaselineHasViolations(
            f"refusing to save a baseline with {violations} scope violation(s) — a violation is a "
            "correctness failure, not a number to bank. Fix the leak, then save."
        )
    with path.open("w") as fh:
        json.dump(metrics, fh, indent=2)


def compare_to_baseline(
    current: dict[str, dict],
    baseline: dict[str, dict],
    threshold: float = REGRESSION_THRESHOLD,
) -> list[str]:
    """Compare *current* metrics to *baseline*.

    Returns a (possibly empty) list of human-readable regression messages.
    Non-empty means the gate should fail.

    Rules
    -----
    * ``hit_rate`` drops > *threshold* relative → regression
    * ``mrr`` drops > *threshold* relative → regression
    * ``violations`` > 0 → regression, **absolutely**, not merely when the count increases

    The violation rule was relative until 2026-07-25 and that was wrong. Scope isolation (R5) is a
    correctness property: a fact from another project appearing in this project's results is a
    defect at any count, and comparing to a recorded number meant the gate's strictness depended
    on the worst config anyone had ever saved. It is now absolute, and :func:`save_baseline`
    refuses to record violations at all, so the two rules cannot disagree.

    This exists because widening ``candidate_multiplier`` to 12x produced this harness's first
    violation — pool width turned out to be a *safety* parameter, and it was defended by nothing
    but someone remembering to read the eval output.
    """
    regressions: list[str] = []
    overall_curr = current.get("OVERALL", {})
    overall_base = baseline.get("OVERALL", {})

    for metric in ("hit_rate", "mrr"):
        base_val = overall_base.get(metric)
        curr_val = overall_curr.get(metric)
        if base_val is None or curr_val is None:
            continue
        if base_val == 0:
            if curr_val < base_val:
                regressions.append(f"OVERALL {metric} regressed: {base_val} → {curr_val}")
            continue
        drop = (base_val - curr_val) / base_val
        if drop > threshold:
            pct = round(drop * 100, 1)
            regressions.append(
                f"OVERALL {metric} regressed {pct}%: {base_val} → {curr_val} "
                f"(threshold {round(threshold * 100)}%)"
            )

    curr_viols = overall_curr.get("violations", 0) or 0
    if curr_viols:
        base_viols = overall_base.get("violations", 0) or 0
        regressions.append(
            f"OVERALL violations: {curr_viols} (baseline {base_viols}) — scope isolation is a "
            "correctness property; any violation fails the gate regardless of the baseline"
        )

    return regressions


# ---------------------------------------------------------------------------
# Engine runner
# ---------------------------------------------------------------------------

async def run_evaluation(engine, cases: list[EvalCase]) -> EvalReport:
    """Score *cases* against *engine*, without the run entering the query log.

    The log is what a held-out eval set gets mined from. If the golden set's own queries landed in
    it, that set would be built from the very cases it exists to be independent of — the harness
    would be grading itself on its own homework. Suppressed here rather than at the call sites so
    a new caller cannot forget, and restored afterwards because the engine outlives this function.
    """
    reader = getattr(engine, "reader", None)
    previous = getattr(reader, "log_queries", None)
    if reader is not None and previous is not None:
        reader.log_queries = False
    try:
        outcomes: list[CaseOutcome] = []
        for case in cases:
            if case.mode == "search":
                results = await engine.search(case.query, limit=case.k)
            else:
                results = await engine.recall(case.query, project_id=case.project_id, limit=case.k)
            outcomes.append(evaluate_case(case, results))
        return EvalReport(outcomes=outcomes)
    finally:
        if reader is not None and previous is not None:
            reader.log_queries = previous
