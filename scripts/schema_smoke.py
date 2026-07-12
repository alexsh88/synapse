"""Validate the Synapse schema against Graphiti.

Ingests a rich, multi-type episode under a project scope using the custom
entity/edge types from synapse.core.schema, then prints the typed entities
(with labels + extracted attributes) and the typed relationships, and finally
a scope-composed search. Proves the knowledge model actually shapes extraction.

    python -m scripts.schema_smoke
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from graphiti_core.nodes import EpisodeType

from synapse.config import settings
from synapse.core.knowledge_engine import build_graphiti
from synapse.core.schema import EDGE_TYPE_MAP, EDGE_TYPES, ENTITY_TYPES, Scope

PROJECT_ID = "synapse-demo"

EPISODE = (
    "For the Acme-Store project, we decided to use SQLite instead of PostgreSQL for "
    "the backend because it's far simpler to ship and the data volume is tiny — a "
    "settled decision made by the planner agent. We also established a convention: "
    "always build backend services with FastAPI. And we learned the hard way that "
    "AdMob must run in test mode during development, otherwise you risk a permanent "
    "account ban — a critical gotcha discovered while setting up Acme-Store."
)

QUERY = "Why did Acme-Store choose SQLite, and what should I watch out for with AdMob?"


async def main() -> int:
    if not settings.anthropic_api_key:
        print("[error] ANTHROPIC_API_KEY missing in .env")
        return 2

    graphiti = build_graphiti()
    try:
        await graphiti.build_indices_and_constraints()
        print(f"[ingest] extracting with custom schema into scope {Scope.project(PROJECT_ID)!r} ...")

        result = await graphiti.add_episode(
            name="acme-store-backend-choices",
            episode_body=EPISODE,
            source=EpisodeType.text,
            source_description="schema validation",
            reference_time=datetime.now(timezone.utc),
            group_id=Scope.project(PROJECT_ID),
            entity_types=ENTITY_TYPES,
            edge_types=EDGE_TYPES,
            edge_type_map=EDGE_TYPE_MAP,
        )

        print(f"\n[entities] {len(result.nodes)} extracted:")
        for n in result.nodes:
            labels = [l for l in n.labels if l != "Entity"] or ["Entity"]
            attrs = {k: v for k, v in (n.attributes or {}).items() if v not in (None, "") and not k.endswith("_embedding")}
            print(f"   • [{'/'.join(labels)}] {n.name}")
            for k, v in attrs.items():
                print(f"        {k}: {v}")

        print(f"\n[relationships] {len(result.edges)} extracted:")
        for e in result.edges:
            print(f"   • ({e.name}) {e.fact}")

        print(f"\n[search] scope={Scope.compose(PROJECT_ID)} query={QUERY!r}")
        hits = await graphiti.search(QUERY, group_ids=Scope.compose(PROJECT_ID), num_results=5)
        for h in hits:
            print(f"   ⇒ {h.fact}")

        print("\n[done] schema drives typed extraction + scoped retrieval ✓")
        return 0 if result.nodes and result.edges and hits else 1
    finally:
        await graphiti.close()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
