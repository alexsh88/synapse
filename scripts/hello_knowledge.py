"""Phase 1 "hello knowledge" smoke test.

Proves the Graphiti graph-data layer connects to Neo4j, that a knowledge node
stores and retrieves, and (with --verify after `docker compose restart`) that it
persists across a restart.

This uses Graphiti's own Neo4jDriver — the same data layer Graphiti rides on —
WITHOUT constructing the full Graphiti() object, which eagerly instantiates an
OpenAI LLM/embedder client and fails with no key. Full episodic ingestion
(add_episode -> LLM extraction + embeddings) is a Phase-2 step, gated on the
embedder-provider decision deferred in docs/research/phase-1-verification.md §6B.

Usage:
    python -m scripts.hello_knowledge          # store + retrieve
    python -m scripts.hello_knowledge --verify  # retrieve only (run after restart)
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone

from graphiti_core.driver.neo4j_driver import Neo4jDriver

from synapse.config import settings

NODE_UUID = "hello-knowledge-0001"
LABEL = "SynapseSmoke"


async def store(driver: Neo4jDriver) -> None:
    await driver.execute_query(
        f"""
        MERGE (n:{LABEL} {{uuid: $uuid}})
        SET n.content = $content,
            n.created_at = $created_at
        RETURN n.uuid AS uuid
        """,
        uuid=NODE_UUID,
        content="Synapse is alive — the first knowledge node.",
        created_at=datetime.now(timezone.utc).isoformat(),
    )


async def fetch(driver: Neo4jDriver):
    result = await driver.execute_query(
        f"""
        MATCH (n:{LABEL} {{uuid: $uuid}})
        RETURN n.uuid AS uuid, n.content AS content, n.created_at AS created_at
        """,
        uuid=NODE_UUID,
    )
    records = result.records
    return records[0] if records else None


async def main(verify_only: bool) -> int:
    driver = Neo4jDriver(
        settings.neo4j_uri,
        settings.neo4j_user,
        settings.neo4j_password,
        settings.neo4j_database,
    )
    try:
        if not verify_only:
            await store(driver)
            print(f"[store ] node '{NODE_UUID}' written to Neo4j ({settings.neo4j_uri})")

        record = await fetch(driver)
        if record is None:
            print(f"[verify] FAIL — node '{NODE_UUID}' not found")
            return 1

        print(
            f"[verify] OK — uuid={record['uuid']!r} "
            f"content={record['content']!r} created_at={record['created_at']}"
        )
        if verify_only:
            print("[persist] node survived the restart ✓")
        return 0
    finally:
        await driver.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synapse hello-knowledge smoke test")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="retrieve only (use after `docker compose restart` to prove persistence)",
    )
    args = parser.parse_args()
    raise SystemExit(asyncio.run(main(args.verify)))
