"""Live end-to-end smoke for retrieval via the KnowledgeEngine facade.

Seeds a few facts, then exercises recall (real Graphiti hybrid search + ranking)
and brief (real Neo4j category queries + Redis cache). Uses scope
``project_retrievetest`` — cleaned up by scripts/clear_test_data.py.

    python -m scripts.retrieve_smoke
"""

from __future__ import annotations

import asyncio

from synapse.config import settings
from synapse.core.knowledge_engine import KnowledgeEngine

PROJECT = "retrievetest"

SEEDS = [
    "For the retrievetest project we decided to use Kafka for the event bus, chosen over "
    "Redis Streams, because we need durable replay at high throughput. A settled decision.",
    "Convention for the retrievetest project: always write integration tests before merging a PR.",
    "Lesson from retrievetest: never commit secrets to the repository — a critical gotcha after "
    "an API key was leaked in a commit.",
]


async def main() -> int:
    if not settings.anthropic_api_key:
        print("[error] ANTHROPIC_API_KEY missing in .env")
        return 2

    async with KnowledgeEngine() as engine:
        print("seeding knowledge:")
        for s in SEEDS:
            r = await engine.remember(s, project_id=PROJECT)
            print(f"   [{r.outcome.value}] {r.knowledge_type} -> {r.scope}")

        print("\nrecall: 'what message bus does retrievetest use and why?'")
        hits = await engine.recall("what message bus does retrievetest use and why?", project_id=PROJECT, limit=3)
        for h in hits:
            print(f"   {h.score:.3f}  {h.fact}")
            print(f"          components={h.components}")

        print("\nbrief (first call — builds + caches):")
        b1 = await engine.brief(PROJECT)
        print(f"   summary: {b1.project_summary}")
        print(f"   conventions: {b1.active_conventions}")
        print(f"   decisions:   {b1.key_decisions}")
        print(f"   lessons:     {b1.relevant_lessons}")
        print(f"   cached={b1.cached}")

        b2 = await engine.brief(PROJECT)
        print(f"\nbrief (second call): cached={b2.cached}")

        ok = bool(hits) and b1.cached is False and b2.cached is True
        print(f"\n[done] live retrieval {'works ✓' if ok else 'UNEXPECTED — review above'}")
        return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
