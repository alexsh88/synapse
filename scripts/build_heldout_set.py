"""Build a frozen, held-out retrieval eval set from queries that were actually asked.

    python -m scripts.build_heldout_set --report-only     # inspect the pool, judge nothing
    python -m scripts.build_heldout_set                    # pool + judge + freeze

Why this exists
---------------
The 52-case golden set was written by hand and has been used while tuning. That makes it a
regression gate: it can say retrieval got worse at cases someone already thought of, and nothing
about whether it generalises. This builds the other kind of set — queries nobody chose, judged by
two models from different families, split so the half used for tuning is never the half used for
measuring, and frozen so a later run cannot quietly re-cut it in the ranker's favour.

The pipeline
------------
1. Read the query log (``synapse.core.query_log``), which records every real recall/search.
2. Deduplicate, then split dev/test by a hash of the query text — deterministic, so re-running
   never reshuffles and a query cannot migrate across the split when the corpus grows.
3. Pool candidates from four genuinely different retrieval configurations (TREC pooling), so the
   ground truth is not defined by the ranker being measured.
4. Grade the pool with two judges, report their agreement, and probe both with decoys.
5. Freeze to JSON with the input hash, so a set built from different queries is visibly different.

This is a *tool*, not a scheduled job. Judging costs real calls; run it deliberately.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from synapse.core.knowledge_engine import KnowledgeEngine
from synapse.core.query_log import read_all
from synapse.core.retrieval_engine import RetrievalEngine
from synapse.core.schema import Scope
from synapse.eval.judge import adversarial_probe, agreement_report, judge_one
from synapse.eval.pooling import Strategy, pool_all, pool_stats

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUT_PATH = Path(__file__).resolve().parents[1] / "synapse" / "eval" / "heldout.json"

#: Share of queries reserved for tuning. The rest is the test half and must never be tuned against.
DEV_SHARE = 0.3

#: Below this the exercise is theatre — the intervals would be wider than any effect worth seeing.
MIN_QUERIES = 30


def split_of(query: str) -> str:
    """Deterministic dev/test assignment from the query text.

    A hash rather than a shuffle: the assignment must survive re-runs, new queries arriving, and
    the set being rebuilt on another machine. Anything that re-randomises is an invitation to
    rebuild until the split flatters the ranker.
    """
    digest = hashlib.sha256(query.strip().lower().encode("utf-8")).digest()
    return "dev" if (digest[0] / 255.0) < DEV_SHARE else "test"


def collect_queries(entries: list[dict[str, Any]]) -> list[tuple[str, str | None]]:
    """Deduplicate logged queries, keeping the first project each was asked under."""
    seen: dict[str, tuple[str, str | None]] = {}
    for e in entries:
        q = (e.get("query") or "").strip()
        # Trivially short queries are keystrokes, not questions; they pool badly and judge worse.
        if len(q) < 8:
            continue
        key = q.lower()
        if key not in seen:
            seen[key] = (q, e.get("project_id"))
    return list(seen.values())


def build_strategies(engine) -> list[Strategy]:
    """Four configurations that differ in WHICH facts become candidates, not just their order.

    Each gets its own RetrievalEngine over the shared searcher, for two reasons. Strategies run
    concurrently, and mutating one engine's tunables in place would have them racing each other.
    And every pooling engine takes ``redis=None`` so this run does not write thousands of synthetic
    queries into the log it just mined — a set that trains on its own construction traffic is
    worthless.
    """
    reader = engine.reader
    base: dict[str, Any] = dict(
        cluster_resolver=reader.cluster_resolver,
        weights=reader.weights,
        mmr_lambda=reader.mmr_lambda,
        max_per_source=reader.max_per_source,
        redis=None,
    )

    def make(**overrides: Any) -> RetrievalEngine:
        return RetrievalEngine(reader.searcher, reader.queries, **{**base, **overrides})

    default = make(candidate_multiplier=reader.candidate_multiplier,
                   min_relevance=reader.min_relevance, rescue_floor=reader.rescue_floor)
    wide = make(candidate_multiplier=reader.candidate_multiplier * 3,
                min_relevance=reader.min_relevance, rescue_floor=reader.rescue_floor)
    # No floor at all: admits the confident junk the floor exists to remove, which is exactly the
    # region where a ranking change might legitimately find something the default never sees.
    unfiltered = make(candidate_multiplier=reader.candidate_multiplier * 2,
                      min_relevance=0.0, rescue_floor=None)
    unscoped = make(candidate_multiplier=reader.candidate_multiplier,
                    min_relevance=reader.min_relevance, rescue_floor=reader.rescue_floor)

    def scoped(eng):
        async def run(query: str, project_id: str | None):
            cluster = eng._cluster_for(project_id)
            return await eng.search(
                query, group_ids=Scope.compose(project_id, cluster=cluster), limit=10,
            )
        return run

    async def all_scopes(query: str, project_id: str | None):
        # group_ids=None means every scope — the only strategy that can surface a fact the
        # project's own scope composition would never reach.
        return await unscoped.search(query, group_ids=None, limit=10)

    return [
        Strategy("default", scoped(default)),
        Strategy("wide", scoped(wide)),
        Strategy("unfiltered", scoped(unfiltered)),
        Strategy("unscoped", all_scopes),
    ]


def make_decoys(pools, limit: int = 12) -> list[tuple[str, str]]:
    """Pair each query with a fact pooled for a DIFFERENT query.

    A decoy has to be plausible prose from the same corpus, or the probe measures nothing harder
    than "can the judge spot lorem ipsum". Facts from another query's pool are real, well-formed,
    and about the wrong thing — which is the failure mode that matters.
    """
    decoys: list[tuple[str, str]] = []
    usable = [p for p in pools if p.candidates]
    for i, pool in enumerate(usable):
        other = usable[(i + len(usable) // 2) % len(usable)]
        if other is pool or not other.candidates:
            continue
        decoys.append((pool.query, other.candidates[0].fact))
        if len(decoys) >= limit:
            break
    return decoys


async def main(report_only: bool, out: Path, min_queries: int) -> int:
    async with KnowledgeEngine() as engine:
        entries = await read_all(engine.reader.redis)
        queries = collect_queries(entries)
        print(f"query log: {len(entries)} records -> {len(queries)} distinct queries")

        if len(queries) < min_queries:
            print(
                f"\n[not enough data] {len(queries)} distinct queries, need {min_queries}.\n"
                "  The log fills as you actually use Synapse — recall/search from any wired\n"
                "  project, or the session-start brief hook, all record here. Come back once it\n"
                "  has accumulated; building a held-out set from a handful of queries would\n"
                "  produce intervals wider than anything it could measure."
            )
            return 3

        splits = {q: split_of(q) for q, _ in queries}
        test = [(q, p) for q, p in queries if splits[q] == "test"]
        dev = [(q, p) for q, p in queries if splits[q] == "dev"]
        print(f"split: {len(dev)} dev (tune against these) / {len(test)} test (never tune)")

        strategies = build_strategies(engine)
        print(f"pooling {len(test)} test queries across {len(strategies)} strategies...")
        pools = await pool_all(test, strategies)
        stats = pool_stats(pools)
        print(json.dumps(stats, indent=2))
        if stats["unique_share"] < 0.05:
            print(
                "\n[warning] almost nothing was found by only one strategy — the four "
                "configurations are behaving identically, so this pool is single-ranker ground "
                "truth with extra steps. Widen them before trusting the set."
            )

        if report_only:
            print("\n--report-only: nothing judged, nothing written.")
            return 0

        total = sum(len(p.candidates) for p in pools)
        print(f"\njudging {total} candidates with two judges ({total * 2} calls)...")
        judged: list[dict[str, Any]] = []
        all_judgements = []
        for pool in pools:
            graded = await asyncio.gather(
                *(judge_one(pool.query, c.uuid, c.fact) for c in pool.candidates)
            )
            all_judgements.extend(graded)
            judged.append({
                "query": pool.query,
                "project_id": pool.project_id,
                "candidates": [
                    {
                        "uuid": c.uuid,
                        "scope": c.scope,
                        "found_by": c.found_by,
                        "grades": j.grades,
                        "consensus": j.consensus,
                    }
                    for c, j in zip(pool.candidates, graded)
                ],
            })

        agreement = agreement_report(all_judgements, "claude", "gemma")
        print("\nagreement between judges:")
        print(json.dumps(agreement, indent=2))

        decoys = make_decoys(pools)
        probe = await adversarial_probe(decoys) if decoys else {"probes": 0, "per_judge": {}}
        print("\nadversarial probe (wrong facts the judges accepted anyway):")
        print(json.dumps(probe, indent=2))

        payload = {
            "version": 1,
            "dev_share": DEV_SHARE,
            # Hash of the test queries: a set built from different traffic is visibly a different
            # set, so a score cannot be quietly compared across two incompatible ones.
            "queries_hash": hashlib.sha256(
                "\n".join(sorted(q for q, _ in test)).encode("utf-8")
            ).hexdigest(),
            "n_dev": len(dev),
            "n_test": len(test),
            "pool_stats": stats,
            "judge_agreement": agreement,
            "adversarial_probe": probe,
            "queries": judged,
        }
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nwrote {out} ({len(judged)} queries, {total} judged candidates)")

        kappa = agreement.get("cohens_kappa")
        if kappa is not None and kappa < 0.4:
            print(
                f"\n[warning] kappa {kappa} is weak agreement. Grades this inconsistent cannot "
                "support a headline number — tighten the rubric before quoting anything from "
                "this set."
            )
        return 0


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a held-out retrieval eval set")
    p.add_argument("--report-only", action="store_true",
                   help="Pool and print statistics without spending judge calls.")
    p.add_argument("--out", type=Path, default=OUT_PATH)
    p.add_argument("--min-queries", type=int, default=MIN_QUERIES)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    raise SystemExit(asyncio.run(main(args.report_only, args.out, args.min_queries)))
