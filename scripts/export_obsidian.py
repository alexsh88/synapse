"""Export the Synapse graph to a read-only Obsidian vault (Phase 12).

    python -m scripts.export_obsidian [out_dir]     # default: exports/obsidian

Open the resulting folder as an Obsidian vault. It's generated + overwritten each run — never
edit it by hand (Neo4j is the source of truth, R3). One-directional; doubles as a readable backup.
"""

from __future__ import annotations

import asyncio
import sys

from synapse.core.knowledge_engine import build_graphiti
from synapse.core.obsidian_export import ObsidianExporter


async def main(out_dir: str) -> int:
    graphiti = build_graphiti()
    try:
        stats = await ObsidianExporter(graphiti, out_dir).export()
        print(f"exported {stats['notes']} notes, {stats['edges']} links -> {stats['out_dir']}")
        for scope, c in sorted(stats["scopes"].items(), key=lambda kv: -kv[1]):
            print(f"  {scope:24} {c}")
        return 0
    finally:
        await graphiti.close()


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "exports/obsidian"
    raise SystemExit(asyncio.run(main(out)))
