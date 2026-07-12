"""Phase 2 first step — a real Graphiti `add_episode` round-trip.

Proves the full Synapse knowledge stack works end-to-end:
  Claude Sonnet 4.6 (extraction)  ->  Neo4j (graph)  ->  BGE-M3 via Ollama (embeddings)
and that the ingested knowledge is retrievable via semantic + graph search.

Modes:
  --check   wiring only: construct Graphiti, build indices, embed a string via
            BGE-M3/Ollama, confirm 1024-dim. Needs NO Claude key. Validates the
            local half (embedder + Neo4j + Graphiti construction).
  (default) full round-trip: add_episode (real Claude extraction) then search.
            Requires ANTHROPIC_API_KEY in .env.

Usage:
  python -m scripts.hello_episode --check
  python -m scripts.hello_episode
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone

from graphiti_core.nodes import EpisodeType

from synapse.config import settings
from synapse.core.knowledge_engine import EMBED_DIM, build_embedder, build_graphiti

GROUP_ID = "synapse-smoke"

EPISODE_TEXT = (
    "Synapse uses BGE-M3 embeddings, run locally via Ollama, for semantic search. "
    "Claude Sonnet 4.6 handles entity extraction. Neo4j is the graph database, and "
    "the whole stack is self-hosted with no OpenAI dependency."
)

QUERY = "What embedding model does Synapse use and how is it run?"


async def check() -> int:
    """Local-only wiring check (no Claude key needed)."""
    embedder = build_embedder()
    vec = await embedder.create(input_data=EPISODE_TEXT)
    assert len(vec) == EMBED_DIM, f"expected {EMBED_DIM}-dim, got {len(vec)}"
    print(f"[embed ] BGE-M3 via Ollama returned {len(vec)}-dim vector ✓")

    graphiti = build_graphiti()
    try:
        await graphiti.build_indices_and_constraints()
        print("[graph ] Graphiti constructed + Neo4j indices/constraints built ✓")
        print(f"[graph ] connected to {settings.neo4j_uri}")
    finally:
        await graphiti.close()
    print("[check ] local half (embedder + graph) is wired correctly.")
    print("[check ] set ANTHROPIC_API_KEY in .env, then run without --check for the full round-trip.")
    return 0


async def full() -> int:
    """Full add_episode -> search round-trip (needs Claude)."""
    if not settings.anthropic_api_key:
        print("[error ] ANTHROPIC_API_KEY is empty in .env — required for Claude extraction.")
        print("[error ] add a key and re-run, or use --check for the local-only wiring test.")
        return 2

    graphiti = build_graphiti()
    try:
        await graphiti.build_indices_and_constraints()
        print("[graph ] indices ready; ingesting episode (Claude extraction + BGE-M3 embedding)...")

        result = await graphiti.add_episode(
            name="synapse-genesis",
            episode_body=EPISODE_TEXT,
            source=EpisodeType.text,
            source_description="phase-2 round-trip test",
            reference_time=datetime.now(timezone.utc),
            group_id=GROUP_ID,
        )
        nodes = getattr(result, "nodes", []) or []
        edges = getattr(result, "edges", []) or []
        print(f"[ingest] episode stored. extracted {len(nodes)} entities, {len(edges)} facts:")
        for n in nodes:
            print(f"           • entity: {n.name}")
        for e in edges:
            print(f"           • fact:   {e.fact}")

        print(f"\n[search] query: {QUERY!r}")
        hits = await graphiti.search(QUERY, group_ids=[GROUP_ID], num_results=5)
        if not hits:
            print("[search] no results — round-trip INCOMPLETE")
            return 1
        for h in hits:
            print(f"           ⇒ {h.fact}")
        print("\n[done  ] real add_episode round-trip works: extract → store → embed → retrieve ✓")
        return 0
    finally:
        await graphiti.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synapse add_episode round-trip")
    parser.add_argument("--check", action="store_true", help="local wiring only (no Claude key)")
    args = parser.parse_args()
    raise SystemExit(asyncio.run(check() if args.check else full()))
