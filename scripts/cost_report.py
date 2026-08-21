"""What Synapse has spent, from its own ledger.

    python -m scripts.cost_report            # summary
    python -m scripts.cost_report --json     # machine-readable

Read-only. Reads the Redis ledger written by ``synapse.core.cost`` as calls happen, so unlike
reconstructing spend from an invoice afterwards, this attributes cost to the operation that
incurred it.

Two honest caveats, both visible in the output rather than buried here. Extraction runs inside
Graphiti, which does not surface token usage back to the caller, so those calls are counted and
attributed by provider but not priced. And the price table in ``synapse/core/cost.py`` is a
default that drifts with provider pricing — check it before quoting a figure anywhere that matters.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from synapse.config import settings
from synapse.core.cost import read_all, summarize

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


async def main(as_json: bool, days: int) -> int:
    if not settings.redis_url:
        print("[error] REDIS_URL is not set — there is no ledger to read.")
        return 2

    import redis.asyncio as aioredis

    client = aioredis.from_url(settings.redis_url, decode_responses=True)
    try:
        entries = await read_all(client)
    finally:
        await client.aclose()

    if not entries:
        print(
            "Ledger is empty. It fills as Synapse makes LLM calls — triage on every write, and\n"
            "extraction attribution through the hybrid client. Nothing recorded before the ledger\n"
            "landed can be recovered."
        )
        return 0

    summary = summarize(entries)
    if as_json:
        print(json.dumps(summary, indent=2))
        return 0

    print(f"Synapse cost ledger — {summary['calls']} calls recorded\n")
    print(f"  priced calls    {summary['priced_calls']:>6}   ${summary['total_cost_usd']:.4f}")
    print(f"  unpriced calls  {summary['unpriced_calls']:>6}   (extraction; Graphiti hides usage)")

    print("\n  by provider")
    for provider, n in sorted(summary["by_provider"].items(), key=lambda kv: -kv[1]):
        print(f"    {provider:<14} {n:>6} calls")

    print("\n  by operation")
    rows = sorted(summary["by_operation"].items(), key=lambda kv: -kv[1]["calls"])
    for op, v in rows:
        unpriced = f"  ({v['unpriced']} unpriced)" if v["unpriced"] else ""
        print(f"    {op:<28} {v['calls']:>6} calls   ${v['cost_usd']:.4f}{unpriced}")

    print(f"\n  last {days} days")
    for day, v in list(summary["by_day"].items())[:days]:
        unpriced = f"  ({v['unpriced']} unpriced)" if v["unpriced"] else ""
        print(f"    {day}   {v['calls']:>5} calls   ${v['cost_usd']:.4f}{unpriced}")

    # The number the extraction_mode decision actually turns on.
    anthropic = summary["by_provider"].get("anthropic", 0)
    total = sum(summary["by_provider"].values())
    if total:
        print(
            f"\n  {anthropic}/{total} calls went to the cloud "
            f"({anthropic / total:.0%}). In hybrid mode that share is the local model's "
            "escalation rate — the cost of its ~14% silent-failure tail, priced."
        )
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Report Synapse's recorded LLM spend")
    p.add_argument("--json", action="store_true", help="Machine-readable summary.")
    p.add_argument("--days", type=int, default=14, help="How many days to list.")
    args = p.parse_args()
    raise SystemExit(asyncio.run(main(args.json, args.days)))
