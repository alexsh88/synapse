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
import sys

from synapse.config import settings
from synapse.core.knowledge_engine import KnowledgeEngine
from synapse.core.retrieval_engine import RankWeights
from synapse.eval import GOLDEN_SET, run_evaluation
from synapse.eval.runner import compare_to_baseline, load_baseline, save_baseline

# The report and regression messages contain non-ASCII (arrows, ellipses) and this script is
# normally run from a Windows console whose default codec is cp1252 — which raised
# UnicodeEncodeError *after* a full eval run, throwing away the results. Same failure class as
# the prompt_recall.py stdin bug: never let an encoding assumption discard real work.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


async def main(
    save_baseline_flag: bool,
    *,
    mmr_lambda: float | None = None,
    weights: RankWeights | None = None,
    no_rescue: bool = False,
    max_per_source: int | None = None,
    candidate_multiplier: int | None = None,
    min_relevance: float | None = None,
    rescue_floor: float | None = None,
) -> int:
    if not settings.anthropic_api_key:
        print("[error] ANTHROPIC_API_KEY missing")
        return 2

    async with KnowledgeEngine() as engine:
        # Ranking overrides for A/B-ing a change without editing code. The engine is already
        # wired, so we just retune the reader in place.
        if mmr_lambda is not None:
            engine.reader.mmr_lambda = mmr_lambda
            print(f"[override] mmr_lambda={mmr_lambda}")
        if min_relevance is not None:
            engine.reader.min_relevance = min_relevance
            print(f"[override] min_relevance={min_relevance}")
        if rescue_floor is not None:
            engine.reader.rescue_floor = rescue_floor or None
            print(f"[override] rescue_floor={engine.reader.rescue_floor}")
        if candidate_multiplier is not None:
            engine.reader.candidate_multiplier = candidate_multiplier
            print(f"[override] candidate_multiplier={candidate_multiplier}")
        if max_per_source is not None:
            engine.reader.max_per_source = max_per_source
            print(f"[override] max_per_source={max_per_source}")
        if no_rescue:
            engine.reader.rescue_floor = None
            print("[override] rescue band OFF (single-threshold floor "
                  f"{engine.reader.min_relevance})")
        if weights is not None:
            engine.reader.weights = weights
            print(
                f"[override] weights rel={weights.relevance} rec={weights.recency} "
                f"conf={weights.confidence} conn={weights.connectivity} "
                f"fusion={weights.fusion}"
            )
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
    parser.add_argument(
        "--mmr-lambda", type=float, default=None,
        help="Override MMR diversity lambda (1.0 = off / pure relevance). Isolates the "
             "diversity pass from the additive ranking weights when A/B-ing a change.",
    )
    parser.add_argument(
        "--min-relevance", type=float, default=None, metavar="F",
        help="Override the similarity floor — the line above which no lexical anchor is needed.",
    )
    parser.add_argument(
        "--rescue-floor", type=float, default=None, metavar="F",
        help="Override the rescue band's hard bottom. 0 disables the band (same as --no-rescue). "
             "Tune this WITH --min-relevance: they are one mechanism with two knobs.",
    )
    parser.add_argument(
        "--candidate-multiplier", type=int, default=None, metavar="N",
        help="Override how many candidates are fetched per requested result. The ranking passes "
             "can only re-order what they are given, so this bounds every quality gain above it.",
    )
    parser.add_argument(
        "--max-per-source", type=int, default=None, metavar="N",
        help="Cap how many results may share one source node (the hub-monopoly cap). "
             "Omit for no cap.",
    )
    parser.add_argument(
        "--no-rescue", action="store_true",
        help="Disable the lexical rescue band, restoring the old single-threshold similarity "
             "floor. This is the like-for-like control for roadmap item 17 — always measure the "
             "OLD config today rather than trusting a stored baseline.",
    )
    parser.add_argument(
        "--weights", default=None, metavar="REL,REC,CONF,CONN[,FUSION]",
        help="Override RankWeights as four or five comma-separated floats, "
             "e.g. --weights 0.70,0.20,0.10,0.10 or 0.35,0.20,0.10,0.10,0.35",
    )
    return parser.parse_args()


def _parse_weights(raw: str | None) -> RankWeights | None:
    if not raw:
        return None
    parts = [float(p) for p in raw.split(",")]
    if len(parts) not in (4, 5):
        raise SystemExit("--weights needs 4 or 5 floats: REL,REC,CONF,CONN[,FUSION]")
    return RankWeights(relevance=parts[0], recency=parts[1], confidence=parts[2],
                       connectivity=parts[3], fusion=parts[4] if len(parts) == 5 else 0.0)


if __name__ == "__main__":
    args = _parse_args()
    raise SystemExit(asyncio.run(main(
        save_baseline_flag=args.save_baseline,
        mmr_lambda=args.mmr_lambda,
        weights=_parse_weights(args.weights),
        no_rescue=args.no_rescue,
        max_per_source=args.max_per_source,
        candidate_multiplier=args.candidate_multiplier,
        min_relevance=args.min_relevance,
        rescue_floor=args.rescue_floor,
    )))
