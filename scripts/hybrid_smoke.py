"""Smoke-test the extraction router end-to-end: one real remember in the configured mode.

    EXTRACTION_MODE=hybrid PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m scripts.hybrid_smoke
    EXTRACTION_MODE=local  ... (full local)        EXTRACTION_MODE=cloud ... (Sonnet)

Writes one fact to a throwaway scope, prints what got extracted, then deletes it.
"""

from __future__ import annotations

import asyncio

from synapse.config import settings
from synapse.core.knowledge_engine import KnowledgeEngine

EPISODE = ("Synapse's extraction router supports a hybrid mode: Gemma3-12B runs locally via Ollama "
           "with strict JSON-schema decoding for bulk writes, and Claude Sonnet 4.6 is the fallback "
           "when the local model fails schema validation. This keeps the pipeline near zero cost.")


async def main() -> int:
    print(f"extraction_mode={settings.extraction_mode}  local={settings.local_extraction_model}")
    async with KnowledgeEngine() as engine:
        r = await engine.remember(EPISODE, project_id="hybridsmoke", source="smoke", force=True)
        print(f"outcome={r.outcome.value} scope={r.scope}")
        print(f"entities ({len(r.entities)}): {r.entities[:8]}")
        print(f"facts ({len(r.facts)}):")
        for f in r.facts[:8]:
            print(f"   - {f[:90]}")
        await engine.graphiti.driver.execute_query(
            "MATCH (n) WHERE n.group_id = 'project_hybridsmoke' DETACH DELETE n")
        print("cleaned up project_hybridsmoke")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
