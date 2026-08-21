#!/usr/bin/env python
"""Fold existing edge names onto the schema vocabulary, and report what is left.

Research §2.2. Measured on the live graph 2026-07-25: **535 distinct edge names over 3,018
edges**. The 11 schema types cover 1,664; 516 extractor-invented names cover 1,354 — and 328 of
those names appear on exactly ONE edge. Typed graph traversal is impossible against that, which
also blocks the structural retrieval lens.

The write pipeline now canonicalizes on write, so the vocabulary stops growing. This script cleans
up what predates that.

**Renaming only, and only where it is safe.** ``canonical_edge_name`` folds a name in only when it
means the same relation in the same direction, so this script never changes graph structure, never
touches endpoints, and never flattens a specific relation into a generic one.

Roadmap item 24 then gave the recurring residual real types (``Uses``, ``DefinedIn``, ``Causes``,
``CausedBy``, ``Fixes``, ``Enforces`` and friends), so most of it now folds legitimately. What still
keeps its own name is what *should*: relations below the volume bar (``TRIGGERS``), project-specific
ones (``PINNED_SECTOR_ETF``), and the ~328 one-off names the extractor invented once each. Note
``Uses`` and ``UsedIn`` remain distinct because they are inverses, as do ``Causes``/``CausedBy`` and
``PartOf``/``Contains`` — no edge ever has its endpoints swapped.

DRY RUN BY DEFAULT. ``--apply`` takes a full backup first and verifies zero loss afterwards (R8).

    python scripts/normalize_edge_names.py                # show the plan + residual report
    python scripts/normalize_edge_names.py --apply        # backup, rename, verify
    python scripts/normalize_edge_names.py --residual 40  # more of the leftover vocabulary
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections import defaultdict

from synapse.config import settings
from synapse.core.backup import BackupService, CurationSafetyError
from synapse.core.schema import EDGE_TYPES, canonical_edge_name
from synapse.db.neo4j_client import direct_graph

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


async def _name_counts(driver) -> list[tuple[str, int]]:
    res = await driver.execute_query(
        "MATCH ()-[e:RELATES_TO]->() RETURN e.name AS name, count(*) AS c ORDER BY c DESC"
    )
    return [(r["name"], int(r["c"])) for r in res.records if r["name"]]


async def run(apply: bool, residual_limit: int) -> int:
    # Direct driver: this script only reads and rewrites properties, so it must not build an
    # embedder or run index DDL. See synapse/db/neo4j_client.py.
    async with direct_graph() as graphiti:
        driver = graphiti.driver
        counts = await _name_counts(driver)
        total_edges = sum(c for _, c in counts)
        print(f"{len(counts)} distinct edge names over {total_edges} edges\n")

        plan: dict[str, list[tuple[str, int]]] = defaultdict(list)
        residual: list[tuple[str, int]] = []
        for name, count in counts:
            canonical = canonical_edge_name(name)
            if canonical != name:
                plan[canonical].append((name, count))
            elif name not in EDGE_TYPES:
                residual.append((name, count))

        affected = sum(c for vs in plan.values() for _, c in vs)
        if plan:
            print("RENAME PLAN (same relation, same direction — zero semantic change):")
            for canonical, variants in sorted(plan.items(), key=lambda kv: -sum(c for _, c in kv[1])):
                pretty = ", ".join(f"{n} ({c})" for n, c in variants)
                print(f"  {canonical:16} <- {pretty}")
            print(f"\n  {affected} edge(s) across {sum(len(v) for v in plan.values())} name(s)\n")
        else:
            print("RENAME PLAN: nothing to do — every name is already canonical.\n")

        # The residual is the actionable finding: these names carry real semantics that the schema
        # does not model, so they must NOT be folded into RelatedTo. Report, never flatten.
        singles = [n for n, c in residual if c == 1]
        print(f"RESIDUAL VOCABULARY: {len(residual)} names, {sum(c for _, c in residual)} edges")
        print(f"  {len(singles)} of them appear on exactly one edge (extractor one-offs).")
        print("  These keep their names. Recurring ones are candidates for a deliberate schema")
        print("  extension (EDGE_TYPES) — flattening them into RelatedTo would destroy meaning.")
        recurring = [(n, c) for n, c in residual if c > 1]
        for name, count in recurring[:residual_limit]:
            print(f"    {count:5}  {name}")
        if len(recurring) > residual_limit:
            print(f"    ... and {len(recurring) - residual_limit} more recurring names "
                  f"(raise --residual to see them)")

        if not apply:
            print("\nDRY RUN — nothing changed. Re-run with --apply to rename.")
            return 0
        if not plan:
            return 0

        backup = BackupService(graphiti, "backups")
        snapshot = await backup.collect()
        path = await backup.snapshot("edge-name-normalization")
        print(f"\nbackup written: {path}")

        renamed = 0
        for canonical, variants in plan.items():
            res = await driver.execute_query(
                """
                MATCH ()-[e:RELATES_TO]->() WHERE e.name IN $olds
                SET e.name_before_canonicalization = e.name, e.name = $new
                RETURN count(e) AS n
                """,
                olds=[n for n, _ in variants], new=canonical,
            )
            n = int(res.records[0]["n"]) if res.records else 0
            renamed += n
            print(f"  {canonical:16} <- {n} edge(s)")

        try:
            check = await backup.verify_no_loss(snapshot)
        except CurationSafetyError as exc:
            print(f"\nZERO-LOSS VIOLATED: {exc}\n  restore from {path}")
            return 1
        print(f"\nrenamed {renamed} edge(s); zero-loss verified "
              f"({check['edges_checked']} edges, {check['nodes_checked']} nodes)")
        after = await _name_counts(driver)
        print(f"distinct edge names: {len(counts)} -> {len(after)}")
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="perform the renames (backup first)")
    parser.add_argument("--residual", type=int, default=20,
                        help="how many recurring residual names to list (default 20)")
    args = parser.parse_args()
    if not settings.neo4j_password:
        print("[error] NEO4J_PASSWORD not set")
        return 2
    return asyncio.run(run(args.apply, args.residual))


if __name__ == "__main__":
    sys.exit(main())
