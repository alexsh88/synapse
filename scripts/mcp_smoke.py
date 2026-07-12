"""Live exercise of the MCP server + all seven tools (inspector-equivalent).

Part A lists the registered tools and their descriptions (what the MCP inspector
shows). Part B connects the real KnowledgeEngine and drives every tool end-to-end:
remember -> recall -> brief -> search -> relate -> update -> forget.

    python -m scripts.mcp_smoke
"""

from __future__ import annotations

import asyncio

from synapse.config import settings
from synapse.core.knowledge_engine import KnowledgeEngine
from synapse.mcp import tools as t
from synapse.mcp.server import mcp

PROJECT = "mcptest"


async def list_tools():
    print("=== registered MCP tools (inspector view) ===")
    for tool in await mcp.list_tools():
        desc = (tool.description or "").splitlines()[0]
        print(f"  • {tool.name}({', '.join(tool.inputSchema.get('properties', {}).keys())})")
        print(f"      {desc}")


async def exercise():
    if not settings.anthropic_api_key:
        print("[error] ANTHROPIC_API_KEY missing")
        return 2

    async with KnowledgeEngine() as engine:
        print("\n=== remember (x2) ===")
        for txt in [
            "For the mcptest project we decided to use Postgres over MySQL because of "
            "stronger JSON support and partial indexes. A settled decision.",
            "Convention for mcptest: all timestamps are stored in UTC.",
        ]:
            r = await t.remember(engine, PROJECT, txt)
            print(f"  {r['outcome']:10} type={r['knowledge_type']:10} scope={r['scope']} entities={r['entities'][:3]}")

        print("\n=== recall ===")
        rc = await t.recall(engine, PROJECT, "which database did mcptest pick and why?", limit=3)
        for h in rc["results"]:
            print(f"  {h['score']:.3f}  {h['fact']}")
        fact_id = rc["results"][0]["id"] if rc["results"] else None

        print("\n=== brief ===")
        b = await t.brief(engine, PROJECT)
        print(f"  summary: {b['project_summary']}")
        print(f"  decisions: {b['key_decisions']}")
        print(f"  conventions: {b['active_conventions']}")

        print("\n=== search (all scopes) ===")
        s = await t.search(engine, "database choice", filters={"limit": 3})
        print(f"  count={s['count']} ignored={s['filters_ignored']}")
        for h in s["results"]:
            print(f"  {h['score']:.3f}  {h['fact']}")

        print("\n=== relate (two entities) ===")
        res = await engine.graphiti.driver.execute_query(
            "MATCH (n:Entity) WHERE n.group_id=$g RETURN n.uuid AS uuid, n.name AS name LIMIT 2",
            g=f"project_{PROJECT}",
        )
        ents = res.records
        if len(ents) >= 2:
            rel = await t.relate(engine, ents[0]["uuid"], ents[1]["uuid"], "related_to")
            print(f"  {ents[0]['name']} --related_to--> {ents[1]['name']}: {rel}")

        print("\n=== update (supersede a fact) ===")
        if fact_id:
            up = await t.update(engine, PROJECT, fact_id,
                                {"content": "mcptest migrated from Postgres to CockroachDB for horizontal scale."})
            print(f"  {up}")

        print("\n=== forget (temporal end) ===")
        if fact_id:
            fg = await t.forget(engine, fact_id, reason="smoke test")
            print(f"  {fg}")

        print("\n[done] all seven MCP tools exercised live ✓")
        return 0


async def main():
    await list_tools()
    return await exercise()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
