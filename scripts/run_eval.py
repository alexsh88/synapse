"""Run the retrieval-quality golden set against live Synapse (Stage 4C).

    python -m scripts.run_eval                # run + compare to baseline (if exists)
    python -m scripts.run_eval --save-baseline  # run + write/overwrite baseline.json

Use before/after a ranking change (RankWeights) to see if quality improved.
Requires the projects to be seeded (scripts/connect_projects_test.py).

Exit codes
----------
0   Metrics reported; no baseline regression (or no baseline on file).
1   Baseline regression detected (hit@k or MRR dropped >5% relative, or violations
    increased).  Review the regression output before merging a ranking change.
2   Configuration error (e.g. missing API key).
"""

from __future__ import annotations

import argparse
import asyncio

from synapse.config import settings
from synapse.core.knowledge_engine import KnowledgeEngine
from synapse.eval import GOLDEN_SET, run_evaluation
from synapse.eval.runner import compare_to_baseline, load_baseline, save_baseline


async def main(save_baseline_flag: bool) -> int:
    if not settings.anthropic_api_key:
        print("[error] ANTHROPIC_API_KEY missing")
        return 2

    async with KnowledgeEngine() as engine:
        report = await run_evaluation(engine, GOLDEN_SET)
        print(report.format())
        metrics = report.metrics()
        overall = metrics["OVERALL"]
        xp = metrics.get("cross_project", {})
        print(
            f"\noverall hit@k={overall['hit_rate']}  MRR={overall['mrr']}"
            f"  precision@k={overall.get('precision_at_k')}  violations={overall.get('violations', 0)}"
            f"  |  cross_project hit@k={xp.get('hit_rate', 'n/a')}"
        )

        if save_baseline_flag:
            from synapse.eval.runner import BASELINE_PATH
            save_baseline(metrics)
            print(f"\n[baseline] saved to {BASELINE_PATH}")
            return 0

        baseline = load_baseline()
        if baseline is None:
            print(
                "\n[baseline] no baseline.json found — reporting only.\n"
                "  Run with --save-baseline to record the current metrics."
            )
            return 0

        regressions = compare_to_baseline(metrics, baseline)
        if regressions:
            print("\n[REGRESSION DETECTED]")
            for msg in regressions:
                print(f"  !! {msg}")
            return 1

        print("\n[baseline] no regression detected.")
        return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synapse retrieval eval harness")
    parser.add_argument(
        "--save-baseline",
        action="store_true",
        help="Write current metrics to baseline.json instead of comparing.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    raise SystemExit(asyncio.run(main(save_baseline_flag=args.save_baseline)))
