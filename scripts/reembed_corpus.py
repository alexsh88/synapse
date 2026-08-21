#!/usr/bin/env python
"""Re-embed the whole corpus onto a new embedding model or dimension.

Research §8, roadmap item 21. The 1024-dim BGE-M3 choice is LOCKED at first ingestion — changing
the embedder or the dimension means rebuilding every vector in the graph. This script exists so
that day is a routine two-minute operation instead of an improvisation.

**Written at 3k edges on purpose.** Measured 2026-07-25: 3,065 fact vectors + 2,241 name vectors =
5,306 embeddings, ~1.6 min end to end at batch 128. At 30k edges the same migration is a long
outage against a graph nobody dares touch. The cost of writing this grows faster than the corpus.

What gets rebuilt, and from what
--------------------------------
Both source texts were verified by **recomputing a sample and comparing cosine to the stored
vector**, not by reading Graphiti's source:

* ``RELATES_TO.fact_embedding`` <- ``r.fact``   (cosine 1.000000)
* ``Entity.name_embedding``     <- ``n.name``   (cosine 1.000000)

The second one matters more than it looks. Every Entity also carries a ``summary``, so "embed the
name and the summary" is the natural assumption — and it is **wrong**: name+summary scores only
0.79-0.91 against the stored vectors. Rebuilding from it would produce 2,241 vectors that look
valid, pass every count and dimension check, and silently degrade node search forever. If you ever
change what Graphiti embeds, re-run ``--verify-source`` before trusting this script again.

Safety (R8)
-----------
* **Dry run by default.** ``--apply`` is required to write anything.
* **Backup first**, via the same ``BackupService`` the curation paths use.
* **Idempotent and resumable**: work is selected by "wrong dimension or missing", so an
  interrupted run is finished by re-running it. Nothing is deleted; vectors are overwritten in
  place, and the source text they derive from is never touched.
* **The vector index is dropped and recreated only when the dimension actually changes** — its
  dimension is baked in at creation, so a dimension change with a stale index leaves every
  similarity query silently failing.

    python scripts/reembed_corpus.py                     # plan only: what would change, and cost
    python scripts/reembed_corpus.py --verify-source     # confirm the source-text assumption holds
    python scripts/reembed_corpus.py --apply --limit 50  # rehearse on 50 items, then verify
    python scripts/reembed_corpus.py --apply             # the real migration
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time

from synapse.core.backup import BackupService

# NOTE: no `direct_graph` here, unlike every other maintenance script. Those only rewrite
# properties, so they deliberately avoid building an embedder; this one's entire job IS embedding.

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Measured on the live corpus: 18.3 ms/text at 128, 25.9 at 64, 53.7 at 16. Bigger batches keep
# helping, but 128 already puts the whole corpus under two minutes and keeps peak VRAM modest.
BATCH = 128

# The only vector index in the graph (Graphiti does not create one for RELATES_TO itself; this one
# was added for dedup + k-NN curation). `Entity.name_embedding` deliberately has none.
FACT_INDEX = "synapse_relates_fact_vec"

_COUNT_EDGES = """
MATCH ()-[r:RELATES_TO]->()
WHERE r.fact IS NOT NULL
  AND (r.fact_embedding IS NULL OR size(r.fact_embedding) <> $dim)
RETURN count(*) AS n
"""
_FETCH_EDGES = """
MATCH ()-[r:RELATES_TO]->()
WHERE r.fact IS NOT NULL
  AND (r.fact_embedding IS NULL OR size(r.fact_embedding) <> $dim)
RETURN r.uuid AS uuid, r.fact AS text
LIMIT $batch
"""
_WRITE_EDGES = """
UNWIND $rows AS row
MATCH ()-[r:RELATES_TO {uuid: row.uuid}]->()
SET r.fact_embedding = row.vec
"""

_COUNT_NODES = """
MATCH (n:Entity)
WHERE n.name IS NOT NULL
  AND (n.name_embedding IS NULL OR size(n.name_embedding) <> $dim)
RETURN count(*) AS n
"""
_FETCH_NODES = """
MATCH (n:Entity)
WHERE n.name IS NOT NULL
  AND (n.name_embedding IS NULL OR size(n.name_embedding) <> $dim)
RETURN n.uuid AS uuid, n.name AS text
LIMIT $batch
"""
_WRITE_NODES = """
UNWIND $rows AS row
MATCH (n:Entity {uuid: row.uuid})
SET n.name_embedding = row.vec
"""

TARGETS = [
    ("fact_embedding (RELATES_TO)", _COUNT_EDGES, _FETCH_EDGES, _WRITE_EDGES),
    ("name_embedding (Entity)", _COUNT_NODES, _FETCH_NODES, _WRITE_NODES),
]


async def _index_dimension(driver) -> int | None:
    """Dimension baked into the fact vector index, or None when the index is absent."""
    res = await driver.execute_query(
        "SHOW VECTOR INDEXES YIELD name, options WHERE name = $n RETURN options",
        n=FACT_INDEX,
    )
    if not res.records:
        return None
    cfg = (res.records[0]["options"] or {}).get("indexConfig", {})
    dim = cfg.get("vector.dimensions")
    return int(dim) if dim is not None else None


async def _recreate_index(driver, dim: int) -> None:
    """Drop and recreate the fact vector index at *dim*.

    The dimension is fixed at creation time, so an index left at the old dimension does not error
    loudly — similarity queries just stop matching. That silence is why this is not optional.
    """
    print(f"  dropping {FACT_INDEX} …")
    await driver.execute_query(f"DROP INDEX {FACT_INDEX} IF EXISTS")
    print(f"  recreating {FACT_INDEX} at dim={dim} …")
    await driver.execute_query(
        f"CREATE VECTOR INDEX {FACT_INDEX} IF NOT EXISTS "
        "FOR ()-[r:RELATES_TO]-() ON (r.fact_embedding) "
        "OPTIONS {indexConfig: {`vector.dimensions`: $dim, "
        "`vector.similarity_function`: 'cosine'}}",
        dim=dim,
    )


async def verify_source(graphiti, driver) -> int:
    """Confirm the stored vectors really are the embedding of the text we would rebuild from.

    Run this after ANY Graphiti upgrade. If it stops printing ~1.0, this script's source-text
    assumption has broken and re-embedding would quietly corrupt search.
    """
    print("source-text verification (expect cosine ~1.0):")
    ok = True
    edges = (await driver.execute_query(
        "MATCH ()-[r:RELATES_TO]->() WHERE r.fact_embedding IS NOT NULL "
        "RETURN r.uuid AS uuid, r.fact AS text LIMIT 5")).records
    nodes = (await driver.execute_query(
        "MATCH (n:Entity) WHERE n.name_embedding IS NOT NULL "
        "RETURN n.uuid AS uuid, n.name AS text LIMIT 5")).records

    for label, rows, cypher in (
        ("edge.fact ", edges,
         "MATCH ()-[r:RELATES_TO {uuid:$u}]->() "
         "RETURN vector.similarity.cosine(r.fact_embedding, $v) AS s"),
        ("node.name ", nodes,
         "MATCH (n:Entity {uuid:$u}) "
         "RETURN vector.similarity.cosine(n.name_embedding, $v) AS s"),
    ):
        if not rows:
            continue
        vecs = await graphiti.embedder.create_batch([r["text"] for r in rows])
        for row, vec in zip(rows, vecs):
            sim = (await driver.execute_query(
                cypher, u=row["uuid"], v=list(vec))).records[0]["s"]
            flag = "" if sim > 0.999 else "   <-- MISMATCH"
            if sim <= 0.999:
                ok = False
            print(f"  {label} cos={sim:.6f}{flag}  {row['text'][:56]}")

    if not ok:
        print("\n[FAIL] stored vectors do not match a re-embedding of their source text.\n"
              "       Do NOT run --apply: find what Graphiti embeds now and update this script.")
        return 1
    print("  OK — sources confirmed")
    return 0


async def _process(graphiti, driver, label, count_q, fetch_q, write_q, dim, apply, limit):
    total = (await driver.execute_query(count_q, dim=dim)).records[0]["n"]
    if limit:
        total = min(total, limit)
    print(f"\n{label}: {total} to rebuild")
    if not total:
        return 0
    if not apply:
        print("  (dry run — nothing written)")
        return total

    done, t0 = 0, time.perf_counter()
    while done < total:
        size = min(BATCH, total - done)
        rows = (await driver.execute_query(fetch_q, dim=dim, batch=size)).records
        if not rows:
            break   # selection is "wrong dim or missing", so an empty page means finished
        vecs = await graphiti.embedder.create_batch([r["text"] for r in rows])
        await driver.execute_query(
            write_q,
            rows=[{"uuid": r["uuid"], "vec": list(v)} for r, v in zip(rows, vecs)],
        )
        done += len(rows)
        rate = done / max(time.perf_counter() - t0, 1e-9)
        print(f"  {done}/{total}  ({rate:.0f}/s)")
    return done


async def _remaining(driver, dim) -> int:
    # A plain loop, not sum(... for ...): an `await` inside a generator expression makes it an
    # async generator, which sum() cannot consume.
    total = 0
    for _, count_q, _, _ in TARGETS:
        total += (await driver.execute_query(count_q, dim=dim)).records[0]["n"]
    return total


async def run(apply: bool, dim: int, limit: int | None, check_source: bool) -> int:
    # A real Graphiti (not direct_graph) — unlike every other maintenance script, this one needs
    # the embedder. It still must not be a full KnowledgeEngine: no write pipeline, no retrieval.
    from synapse.core.knowledge_engine import build_graphiti

    graphiti = build_graphiti()
    driver = graphiti.driver
    try:
        if check_source:
            return await verify_source(graphiti, driver)

        index_dim = await _index_dimension(driver)
        print(f"target dimension : {dim}")
        print(f"index dimension  : {index_dim}  ({FACT_INDEX})")
        pending = await _remaining(driver, dim)
        print(f"vectors to rebuild: {pending}")
        if pending:
            print(f"estimated time   : ~{pending * 0.0183 / 60:.1f} min "
                  f"(18.3 ms/text measured at batch {BATCH})")

        if not apply:
            for label, count_q, _, _ in TARGETS:
                n = (await driver.execute_query(count_q, dim=dim)).records[0]["n"]
                print(f"  {label}: {n}")
            print("\nDRY RUN — re-run with --apply to write. "
                  "Run --verify-source first if Graphiti was upgraded.")
            return 0

        if not pending and index_dim == dim:
            print("\nnothing to do — every vector already matches the target dimension.")
            return 0

        print("\nbacking up first (R8) …")
        backup = BackupService(graphiti, "backups")
        snapshot = await backup.collect()
        print(f"  snapshot: {len(snapshot.edges)} edges, {len(snapshot.node_uuids)} nodes")

        for target in TARGETS:
            await _process(graphiti, driver, *target, dim=dim, apply=True, limit=limit)

        # Index last: recreating it before the vectors are rebuilt would index the old ones.
        if index_dim != dim:
            print(f"\ndimension changed ({index_dim} -> {dim}); rebuilding the vector index")
            await _recreate_index(driver, dim)
        else:
            print(f"\ndimension unchanged ({dim}); leaving {FACT_INDEX} in place")

        # Zero-loss, the same gate the curation paths use. Re-embedding only overwrites a
        # property, so losing an edge would mean something went very wrong — which is exactly
        # when a check earns its keep (R8).
        await backup.verify_no_loss(snapshot)
        print("  zero-loss verified: every edge and node from the snapshot still resolves")

        left = await _remaining(driver, dim)
        print(f"\nverification: {left} vector(s) still not at dim={dim}")
        if left and not limit:
            print("[FAIL] re-run to finish (the job is resumable); do not consider this complete")
            return 1
        if limit:
            print("(rehearsal used --limit, so a remainder is expected)")
        print("[OK] re-embed complete. Now re-run scripts/run_eval.py — a dimension change "
              "invalidates the stored baseline, so expect to re-measure, not to match.")
        return 0
    finally:
        await graphiti.close()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Re-embed the corpus (research §8, item 21)")
    p.add_argument("--apply", action="store_true",
                   help="Actually write. Without this, plan only.")
    p.add_argument("--dim", type=int, default=1024,
                   help="Target embedding dimension (default 1024, the locked BGE-M3 value).")
    p.add_argument("--limit", type=int, default=None,
                   help="Rehearse on at most N items per target before committing to a full run.")
    p.add_argument("--verify-source", action="store_true",
                   help="Check that stored vectors match a re-embedding of their source text. "
                        "Run after any Graphiti upgrade, before --apply.")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    raise SystemExit(asyncio.run(
        run(args.apply, args.dim, args.limit, args.verify_source)
    ))
