"""Live end-to-end smoke for the write pipeline.

Exercises the REAL collaborators the unit tests fake out: Claude Haiku triage,
the Neo4j cosine-similarity dedup query, and Graphiti storage. Stores a decision,
then a restatement of it (expect DUPLICATE), then noise (expect REJECTED).

    python -m scripts.write_smoke
"""

from __future__ import annotations

import asyncio

from synapse.config import settings
from synapse.core.knowledge_engine import build_graphiti
from synapse.core.write_pipeline import build_write_pipeline

PROJECT = "writetest"

DECISION = (
    "For the writetest project, we decided to use Kafka for the event bus because we "
    "need durable replay and very high throughput. Redis Streams was considered but "
    "rejected for lack of durable replay at scale."
)
RESTATEMENT = "writetest uses Kafka as its event bus, chosen over Redis Streams for durable replay."
NOISE = "hmm ok let me just check that real quick, one sec"


async def main() -> int:
    if not settings.anthropic_api_key:
        print("[error] ANTHROPIC_API_KEY missing in .env")
        return 2

    graphiti = build_graphiti()
    try:
        await graphiti.build_indices_and_constraints()
        pipeline = build_write_pipeline(graphiti)

        print("\n1) store a fresh decision:")
        r1 = await pipeline.remember(DECISION, project_id=PROJECT)
        print(f"   outcome={r1.outcome.value} scope={r1.scope} type={r1.knowledge_type} "
              f"entities={r1.entities}")

        print("\n2) restate the same decision (expect duplicate):")
        r2 = await pipeline.remember(RESTATEMENT, project_id=PROJECT)
        print(f"   outcome={r2.outcome.value} duplicate_of={r2.duplicate_of} reason={r2.reason!r}")

        print("\n3) send noise (expect rejected by write-trigger):")
        r3 = await pipeline.remember(NOISE, project_id=PROJECT)
        print(f"   outcome={r3.outcome.value} reason={r3.reason!r}")

        ok = (
            r1.outcome.value == "stored"
            and r2.outcome.value == "duplicate"
            and r3.outcome.value == "rejected"
        )
        print(f"\n[done] live write pipeline {'works ✓' if ok else 'UNEXPECTED — review above'}")
        return 0 if ok else 1
    finally:
        await graphiti.close()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
