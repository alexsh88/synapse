#!/usr/bin/env python
"""Refile project-specific facts that are sitting in ``global`` (research §5.3).

The write-time gate (roadmap item 15) stops NEW pollution. This cleans up what predates it.

``global`` is the one scope retrieval composes for EVERY project, and the ``UserPromptSubmit`` hook
injects it into every prompt in every project — so a project-specific fact there is noise eleven
times over. Measured 2026-07-25: of 137 active global facts, **41 name exactly one project** and a
further 5 name several projects that all sit in one cluster. Typical case::

    "The decision to use ib_async instead of ib_insync applies to the Acme-Sim project"
    "The decision to use ib_async instead of ib_insync applies to the Acme-API project"

That is one piece of trading-domain knowledge, stored twice, in the worst possible scope.

The rule comes from ``synapse.core.write_pipeline.better_scope_than_global`` — deliberately the
SAME function the gate uses. If the two could drift, this migration would move facts the gate would
then keep re-admitting.

**Edges only.** Retagging the endpoint *nodes* is not safe in general: all 32 endpoint nodes are
global and are shared with edges in other scopes, so moving a node would silently change the scope
of unrelated facts. Facts are what retrieval serves (``recall`` searches edges, and the hook injects
facts), so retagging edges fixes the pollution consumers actually see. The node situation is
reported, never silently changed.

Nothing is deleted and no endpoints move, so uuids are preserved and ``verify_no_loss`` applies.
DRY RUN BY DEFAULT.

    python scripts/refile_global_facts.py            # show the plan
    python scripts/refile_global_facts.py --apply     # backup, retag, verify
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections import Counter

from synapse.config import settings
from synapse.core.backup import BackupService, CurationSafetyError
from synapse.core.registry import all_projects, cluster_of
from synapse.core.schema import GLOBAL_SCOPE
from synapse.core.write_pipeline import better_scope_than_global
from synapse.db.neo4j_client import direct_graph

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


async def run(apply: bool, show: int) -> int:
    projects = list(all_projects())
    async with direct_graph() as graph:
        driver = graph.driver
        result = await driver.execute_query(
            """
            MATCH (a:Entity)-[e:RELATES_TO]->(b:Entity)
            WHERE e.group_id = $glob
              AND e.invalid_at IS NULL AND coalesce(e.archived, false) = false
            RETURN e.uuid AS uuid, e.fact AS fact, a.uuid AS a_uuid, b.uuid AS b_uuid
            """,
            glob=GLOBAL_SCOPE,
        )
        rows = [(r["uuid"], r["fact"] or "", r["a_uuid"], r["b_uuid"]) for r in result.records]

        plan: list[tuple[str, str, str]] = []   # (uuid, target_scope, fact)
        for uuid, fact, _a, _b in rows:
            target = better_scope_than_global(fact, projects, cluster_of)
            if target:
                plan.append((uuid, target, fact))

        print(f"active global facts: {len(rows)}")
        print(f"project-specific (to refile): {len(plan)}")
        if not plan:
            print("nothing to do — global holds no project-specific facts.")
            return 0
        print("\nBY TARGET SCOPE:")
        for target, n in Counter(t for _u, t, _f in plan).most_common():
            print(f"  {target:28} {n}")

        print(f"\nSAMPLE (first {show}):")
        for uuid, target, fact in plan[:show]:
            print(f"  -> {target:26} {fact[:88]}")

        # The endpoint nodes stay put, on purpose. Say so rather than let it look like an oversight.
        moving_uuids = {u for u, _t, _f in plan}
        node_ids = sorted({
            n for (uuid, _fact, a, b) in rows if uuid in moving_uuids
            for n in (a, b) if n
        })
        node_res = await driver.execute_query(
            """
            MATCH (n:Entity) WHERE n.uuid IN $ids AND n.group_id = $glob
            RETURN count(n) AS n
            """,
            ids=node_ids, glob=GLOBAL_SCOPE,
        )
        stuck = int(node_res.records[0]["n"]) if node_res.records else 0
        print(f"\nENDPOINT NODES LEFT IN global: {stuck}")
        print("  Not retagged on purpose — these nodes are shared with edges in other scopes, so")
        print("  moving one would silently change the scope of unrelated facts. Facts are what")
        print("  retrieval serves, so refiling edges fixes what consumers actually see.")

        if not apply:
            print("\nDRY RUN — nothing changed. Re-run with --apply to refile.")
            return 0

        backup = BackupService(graph, "backups")
        snapshot = await backup.collect()
        path = await backup.snapshot("refile-global-facts")
        print(f"\nbackup written: {path}")

        moved = 0
        for target in sorted({t for _u, t, _f in plan}):
            uuids = [u for u, t, _f in plan if t == target]
            res = await driver.execute_query(
                """
                UNWIND $uuids AS uuid
                MATCH ()-[e:RELATES_TO {uuid: uuid}]->()
                SET e.group_id = $target, e.scope_refiled_from = $glob
                RETURN count(e) AS n
                """,
                uuids=uuids, target=target, glob=GLOBAL_SCOPE,
            )
            n = int(res.records[0]["n"]) if res.records else 0
            moved += n
            print(f"  {target:28} <- {n} fact(s)")

        try:
            check = await backup.verify_no_loss(snapshot)
        except CurationSafetyError as exc:
            print(f"\nZERO-LOSS VIOLATED: {exc}\n  restore from {path}")
            return 1
        after = await driver.execute_query(
            """
            MATCH ()-[e:RELATES_TO]->() WHERE e.group_id = $glob
              AND e.invalid_at IS NULL AND coalesce(e.archived, false) = false
            RETURN count(e) AS n
            """,
            glob=GLOBAL_SCOPE,
        )
        remaining = int(after.records[0]["n"]) if after.records else 0
        print(f"\nrefiled {moved} fact(s); zero-loss verified "
              f"({check['edges_checked']} edges, {check['nodes_checked']} nodes)")
        print(f"active global facts: {len(rows)} -> {remaining}")
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="perform the refiling (backup first)")
    parser.add_argument("--show", type=int, default=12, help="sample rows to print (default 12)")
    args = parser.parse_args()
    if not settings.neo4j_password:
        print("[error] NEO4J_PASSWORD not set")
        return 2
    return asyncio.run(run(args.apply, args.show))


if __name__ == "__main__":
    sys.exit(main())
