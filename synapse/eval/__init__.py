"""Synapse retrieval-quality evaluation (Stage 4C).

A golden-set harness to measure whether `recall`/`search` surface the *right*
knowledge — the core value of the system (R7). Run before/after ranking changes.
"""

from synapse.eval.cases import GOLDEN_SET, EvalCase
from synapse.eval.runner import (
    CaseOutcome,
    EvalReport,
    compare_to_baseline,
    evaluate_case,
    load_baseline,
    run_evaluation,
    save_baseline,
)

__all__ = [
    "GOLDEN_SET",
    "EvalCase",
    "CaseOutcome",
    "EvalReport",
    "evaluate_case",
    "run_evaluation",
    "load_baseline",
    "save_baseline",
    "compare_to_baseline",
]
