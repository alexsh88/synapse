#!/usr/bin/env python
"""Stamp `content_hash` on episodes ingested before the exact-duplicate guard existed.

Research §2.5. The write pipeline's step 4a is a deterministic, race-free exact-duplicate check:
sha256 over whitespace-normalized content, matched against already-stored episodes in the same
scope. It exists because the vector path compares content-embeddings to fact-embeddings (different
provenance) and is check-then-act racy under concurrent writes, so it cannot reliably catch an
identical re-submit.

That guard only works for episodes carrying the hash, and it was added late: measured 2026-07-25,
**30 of 310 episodes had one**. The other 90% were unprotected — re-submitting their exact content
would store a second copy.

Writing a hash cannot lose knowledge: it only adds a property. Still DRY RUN BY DEFAULT, and
`--apply` verifies zero loss afterwards, because R8 applies to every migration.

    python scripts/backfill_content_hashes.py           # report coverage + the plan
    python scripts/backfill_content_hashes.py --apply    # stamp, then verify
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from synapse.config import settings
from synapse.core.backup import BackupService, CurationSafetyError
from synapse.core.write_pipeline import _content_hash
from synapse.db.neo4j_client import direct_graph

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_BATCH = 200


async def _coverage(driver) -> tuple[int, int]:
    res = await driver.execute_query(
        "MATCH (e:Episodic) RETURN count(e) AS total, count(e.content_hash) AS hashed"
    )
    r = res.records[0]
    return int(r["total"]), int(r["hashed"])


async def run(apply: bool) -> int:
    # Direct driver: this script only reads and rewrites properties, so it must not build an
    # embedder or run index DDL. See synapse/db/neo4j_client.py.
    async with direct_graph() as graphiti:
        driver = graphiti.driver
        total, hashed = await _coverage(driver)
        missing = total - hashed
        print(f"episodes: {total}, with content_hash: {hashed}, missing: {missing}")
        if missing == 0:
            print("nothing to do — every episode is already stamped.")
            return 0

        res = await driver.execute_query(
            """
            MATCH (e:Episodic)
            WHERE e.content_hash IS NULL AND e.content IS NOT NULL
            RETURN e.uuid AS uuid, e.content AS content, e.group_id AS scope
            """
        )
        rows = [(r["uuid"], r["content"], r["scope"]) for r in res.records]
        print(f"stampable now (have content): {len(rows)}")
        if len(rows) < missing:
            # No silent gaps: say which ones cannot be stamped and why.
            print(f"  NOTE: {missing - len(rows)} episode(s) have no `content` property and cannot "
                  f"be hashed; they stay unprotected.")

        # Report collisions the backfill will reveal: identical content already stored twice in one
        # scope. Those are pre-existing exact duplicates the guard would have blocked.
        seen: dict[tuple[str, str], int] = {}
        for _uuid, content, scope in rows:
            key = (scope or "", _content_hash(content))
            seen[key] = seen.get(key, 0) + 1
        collisions = {k: n for k, n in seen.items() if n > 1}
        if collisions:
            print(f"  {len(collisions)} content hash(es) occur more than once within a scope — "
                  f"pre-existing exact duplicates ({sum(collisions.values())} episodes).")
            print("  Stamping is still correct: the hash records what the content IS. Dedup of the "
                  "existing copies is the consolidation engine's job, not this script's.")

        if not apply:
            print("\nDRY RUN — nothing changed. Re-run with --apply to stamp.")
            return 0

        backup = BackupService(graphiti, "backups")
        snapshot = await backup.collect()
        path = await backup.snapshot("content-hash-backfill")
        print(f"\nbackup written: {path}")

        stamped = 0
        for start in range(0, len(rows), _BATCH):
            batch = [
                {"uuid": uuid, "hash": _content_hash(content)}
                for uuid, content, _scope in rows[start : start + _BATCH]
            ]
            await driver.execute_query(
                """
                UNWIND $batch AS b
                MATCH (e:Episodic {uuid: b.uuid})
                SET e.content_hash = b.hash
                """,
                batch=batch,
            )
            stamped += len(batch)
            print(f"  stamped {stamped}/{len(rows)}")

        try:
            check = await backup.verify_no_loss(snapshot)
        except CurationSafetyError as exc:
            print(f"\nZERO-LOSS VIOLATED: {exc}\n  restore from {path}")
            return 1
        total_after, hashed_after = await _coverage(driver)
        print(f"\ncoverage: {hashed}/{total} -> {hashed_after}/{total_after}; zero-loss verified "
              f"({check['edges_checked']} edges, {check['nodes_checked']} nodes)")
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="stamp the hashes (backup first)")
    args = parser.parse_args()
    if not settings.neo4j_password:
        print("[error] NEO4J_PASSWORD not set")
        return 2
    return asyncio.run(run(args.apply))


if __name__ == "__main__":
    sys.exit(main())
