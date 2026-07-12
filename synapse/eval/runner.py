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
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from synapse.eval.cases import EvalCase

BASELINE_PATH = Path(__file__).parent / "baseline.json"
REGRESSION_THRESHOLD = 0.05   # 5 % relative drop triggers gate failure


class CaseOutcome(BaseModel):
    id: str
    category: str
    hit: bool
    rank: int | None = None          # 1-based rank of first expected hit, else None
    top_fact: str = ""
    note: str = ""
    precision_at_k: float | None = None   # None when no positive expectation
    violations: int = 0                   # count of must_not_match hits in top-k


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

    return CaseOutcome(
        id=case.id,
        category=case.category,
        hit=hit,
        rank=hit_rank,
        top_fact=top_fact,
        note=note,
        precision_at_k=precision,
        violations=violations,
    )


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
            prec_group = [o for o in group if o.precision_at_k is not None]
            avg_prec = (sum(o.precision_at_k for o in prec_group) / len(prec_group)
                        if prec_group else None)
            total_violations = sum(o.violations for o in group)
            out[cat] = {
                "n": n,
                "hits": hits,
                "hit_rate": round(hits / n, 3) if n else 0.0,
                "mrr": round(mrr, 3),
                "precision_at_k": round(avg_prec, 4) if avg_prec is not None else None,
                "violations": total_violations,
            }
        return out

    def format(self) -> str:
        m = self.metrics()
        lines = ["Retrieval quality (golden set):", ""]
        lines.append(
            f"  {'category':<16} {'n':>3} {'hits':>5} {'hit@k':>7} {'MRR':>6}"
            f" {'prec@k':>8} {'violations':>11}"
        )
        for cat, v in m.items():
            prec = f"{v['precision_at_k']:.4f}" if v["precision_at_k"] is not None else "   n/a"
            lines.append(
                f"  {cat:<16} {v['n']:>3} {v['hits']:>5} {v['hit_rate']:>7}"
                f" {v['mrr']:>6} {prec:>8} {v['violations']:>11}"
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


def save_baseline(metrics: dict[str, dict], path: Path = BASELINE_PATH) -> None:
    """Persist *metrics* to *path* as the new baseline."""
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
    * ``violations`` increases (any amount) → regression
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

    base_viols = overall_base.get("violations", 0) or 0
    curr_viols = overall_curr.get("violations", 0) or 0
    if curr_viols > base_viols:
        regressions.append(
            f"OVERALL violations increased: {base_viols} → {curr_viols}"
        )

    return regressions


# ---------------------------------------------------------------------------
# Engine runner
# ---------------------------------------------------------------------------

async def run_evaluation(engine, cases: list[EvalCase]) -> EvalReport:
    outcomes: list[CaseOutcome] = []
    for case in cases:
        if case.mode == "search":
            results = await engine.search(case.query, limit=case.k)
        else:
            results = await engine.recall(case.query, project_id=case.project_id, limit=case.k)
        outcomes.append(evaluate_case(case, results))
    return EvalReport(outcomes=outcomes)
