"""TREC-style candidate pooling — how to build a gold set without judging the whole corpus.

The problem: a held-out eval set needs to know, for each query, which facts are actually relevant.
Judging all ~4,600 facts per query is impossible, and judging only what the *current* ranker returns
bakes today's ranker into tomorrow's ground truth — the set would score any change as a regression
purely because it never saw what the new ranker found.

Pooling is the standard answer, from TREC. Run several genuinely different retrieval configurations,
take the **union** of each one's top-k, and judge only that pool. A fact no reasonable configuration
surfaces is treated as non-relevant. The pool is therefore ~15-40 items per query instead of
thousands, and it is not biased toward any single configuration, because every configuration in the
pool contributed to it.

The strategies must actually differ or the pool is one ranker wearing hats. The ones here vary
candidate width, the similarity floor, and scope breadth, which are the three knobs that change
*which* facts become candidates rather than merely how they are ordered.

Nothing here calls an LLM. Judging the pool is ``synapse.eval.judge``'s job; this module only
decides what gets judged.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any, NamedTuple

logger = logging.getLogger("synapse.eval.pooling")

#: How deep each strategy contributes. TREC used 100 with dozens of systems; with a handful of
#: strategies over a few thousand facts, 10 already produces pools in the 15-40 range, and every
#: extra rank is a judgement someone has to pay for.
POOL_DEPTH = 10


class Candidate(NamedTuple):
    """One pooled fact, plus which strategies surfaced it and how high."""

    uuid: str
    fact: str
    scope: str | None
    #: Strategy name -> best (1-based) rank that strategy gave it. Kept because a fact only one
    #: wide-net strategy found at rank 10 is a different kind of candidate from one everything
    #: ranked first, and the judge's disagreements cluster in the former.
    found_by: dict[str, int]

    @property
    def strategy_count(self) -> int:
        return len(self.found_by)

    @property
    def best_rank(self) -> int:
        return min(self.found_by.values())


class PooledQuery(NamedTuple):
    query: str
    project_id: str | None
    candidates: list[Candidate]


#: A strategy is just a named way of getting results for a query. Taking a callable rather than an
#: engine keeps this module free of retrieval internals — and lets the tests pool over lists.
SearchFn = Callable[[str, str | None], Awaitable[list[Any]]]


class Strategy(NamedTuple):
    name: str
    run: SearchFn


async def pool_query(
    query: str,
    project_id: str | None,
    strategies: list[Strategy],
    *,
    depth: int = POOL_DEPTH,
) -> PooledQuery:
    """Run every strategy for one query and union their top-*depth* results."""
    results = await asyncio.gather(
        *(s.run(query, project_id) for s in strategies), return_exceptions=True
    )

    merged: dict[str, Candidate] = {}
    for strategy, res in zip(strategies, results):
        if isinstance(res, BaseException):
            # One strategy failing must not silently shrink the pool without saying so — a quiet
            # pool is how a gold set ends up biased toward whichever strategies happened to work.
            logger.warning("pooling strategy %r failed: %s", strategy.name, str(res)[:160])
            continue
        for rank, item in enumerate(res[:depth], start=1):
            uuid = getattr(item, "uuid", None)
            if not uuid:
                continue
            existing = merged.get(uuid)
            if existing is None:
                merged[uuid] = Candidate(
                    uuid=uuid,
                    fact=getattr(item, "fact", "") or "",
                    scope=getattr(item, "scope", None),
                    found_by={strategy.name: rank},
                )
            else:
                # Keep the best rank each strategy achieved; a strategy cannot vote twice.
                prior = existing.found_by.get(strategy.name)
                if prior is None or rank < prior:
                    existing.found_by[strategy.name] = rank

    # Most-corroborated first, then best rank. This is presentation order for the judge, not a
    # relevance claim — but showing the strongly-corroborated candidates first makes a human
    # spot-check of the top of the pool worth more per minute.
    candidates = sorted(
        merged.values(), key=lambda c: (-c.strategy_count, c.best_rank, c.uuid)
    )
    return PooledQuery(query=query, project_id=project_id, candidates=candidates)


async def pool_all(
    queries: list[tuple[str, str | None]],
    strategies: list[Strategy],
    *,
    depth: int = POOL_DEPTH,
    concurrency: int = 4,
) -> list[PooledQuery]:
    """Pool many queries, bounded so a pooling run cannot saturate the local embedder."""
    sem = asyncio.Semaphore(concurrency)

    async def one(q: str, pid: str | None) -> PooledQuery:
        async with sem:
            return await pool_query(q, pid, strategies, depth=depth)

    return list(await asyncio.gather(*(one(q, pid) for q, pid in queries)))


def pool_stats(pools: list[PooledQuery]) -> dict[str, Any]:
    """Summary a human should read before paying to judge the pool.

    ``unique_to_one_strategy`` is the number that says whether pooling was worth doing: if it is
    near zero the strategies are not actually different and the pool is single-ranker ground truth
    with extra steps.
    """
    sizes = [len(p.candidates) for p in pools]
    unique = sum(1 for p in pools for c in p.candidates if c.strategy_count == 1)
    total = sum(sizes)
    return {
        "queries": len(pools),
        "candidates_total": total,
        "candidates_per_query_mean": round(total / len(pools), 1) if pools else 0.0,
        "candidates_per_query_min": min(sizes) if sizes else 0,
        "candidates_per_query_max": max(sizes) if sizes else 0,
        "unique_to_one_strategy": unique,
        "unique_share": round(unique / total, 3) if total else 0.0,
    }
