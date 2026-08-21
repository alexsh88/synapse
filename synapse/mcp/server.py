"""Synapse MCP server (plan Part 6).

Exposes the cross-project knowledge brain to Claude Code over MCP (stdio). Each
connected project's `.mcp.json` sets ``SYNAPSE_PROJECT_ID``; tools default their
scope to that project. One ``KnowledgeEngine`` is connected for the process
lifetime via the FastMCP lifespan.

Run:  python -m synapse.mcp.server

Logs go to stderr only — stdout is reserved for the MCP protocol (never print()).
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Any

from mcp.server.fastmcp import FastMCP

from synapse.core.knowledge_engine import KnowledgeEngine
from synapse.mcp import tools as t

logger = logging.getLogger("synapse.mcp")

# Scope resolution: the project this server instance speaks for (None => global-only).
PROJECT_ID = os.environ.get("SYNAPSE_PROJECT_ID") or None

# Write attribution (roadmap item 13). This process exists to serve Claude Code over MCP, so it can
# name the writer without being told. setdefault, not assignment: a project's .mcp.json or the shell
# may already have set something more specific, and that should win.
os.environ.setdefault("SYNAPSE_AGENT", "claude-code")

# Single connected engine for the process (set in lifespan).
_engine: KnowledgeEngine | None = None


@asynccontextmanager
async def lifespan(_server: FastMCP):
    global _engine
    logger.info("Synapse MCP starting (project=%s)", PROJECT_ID or "<global>")
    _engine = await KnowledgeEngine().connect()
    logger.info("Knowledge engine connected.")
    try:
        yield {"engine": _engine}
    finally:
        await _engine.close()
        _engine = None
        logger.info("Synapse MCP stopped.")


mcp = FastMCP(
    "synapse",
    instructions=(
        "Synapse is the shared, cross-project knowledge brain. At session start call "
        "brief() to load this project's context. Use remember() when a decision is made "
        "(with rationale), a convention is established, a lesson/gotcha is learned, or "
        "research concludes. Use recall()/search() before solving a problem to check what "
        "was learned before (possibly in another project). Do NOT remember raw chatter, "
        "intermediate reasoning, or debug scratch — Synapse filters noise but keep it clean."
    ),
    lifespan=lifespan,
)


async def _safe(awaitable) -> Any:
    """Run a tool body, converting any exception into a structured error.

    MCP tools must not raise through the transport (that would drop the
    connection); we return ``{"error": ...}`` and log the traceback to stderr.
    """
    try:
        return await awaitable
    except Exception as exc:  # noqa: BLE001 - surface, don't crash the connection
        logger.exception("tool failed")
        return {"error": f"{type(exc).__name__}: {exc}"}


@mcp.tool()
async def remember(content: str, type: str | None = None, scope: str | None = None,
                   relationships: str | None = None) -> dict:
    """Store a durable piece of knowledge (decision, convention, lesson, research, pattern, tool).

    Noise (chatter, scratch, reasoning) is filtered out automatically. `type` is
    auto-detected if omitted. `scope` defaults to this project; pass scope="global"
    for knowledge that applies across all projects. `relationships` is an optional
    free-text hint; links are otherwise auto-extracted.
    """
    return await _safe(t.remember(_engine, PROJECT_ID, content, type=type, scope=scope, relationships=relationships))


@mcp.tool()
async def recall(query: str, scope: str | None = None, limit: int = 10, as_of: str | None = None) -> dict:
    """Retrieve relevant knowledge for this project (global + project scope, ranked).

    `scope`="global" restricts to global knowledge. `as_of` is an ISO timestamp for
    point-in-time ("what did we think on 2026-03-01") queries.
    """
    return await _safe(t.recall(_engine, PROJECT_ID, query, scope=scope, limit=limit, as_of=as_of))


@mcp.tool()
async def brief(project_id: str | None = None) -> dict:
    """Session-start briefing: project summary, active conventions, key decisions, lessons, cross-project knowledge.

    Call this at the start of a session. Defaults to this project (SYNAPSE_PROJECT_ID).
    """
    return await _safe(t.brief(_engine, PROJECT_ID, project_id=project_id))


@mcp.tool()
async def remember_runbook(
    name: str, steps: list[str], purpose: str | None = None,
    prerequisites: str | None = None, scope: str | None = None, verified: bool = True,
) -> dict:
    """Store an ordered, executable procedure — "how to do X here".

    Use this instead of remember() whenever the knowledge is a SEQUENCE: a deploy runbook, a
    debugging procedure, a pre-release checklist, the steps to wire a new service. remember()
    takes prose and extracts entities from it, which loses the ordering; this keeps the steps
    exactly as given.

    `steps` is an ordered list, one action per item. Set verified=false if you are recording a
    procedure you have not just run yourself.
    """
    return await _safe(t.remember_runbook(
        _engine, PROJECT_ID, name, steps, purpose=purpose,
        prerequisites=prerequisites, scope=scope, verified=verified,
    ))


@mcp.tool()
async def runbooks(project_id: str | None = None, limit: int = 20) -> dict:
    """List the procedures available to this project, with their steps.

    Call before doing anything operational (deploying, releasing, debugging a recurring failure)
    — the sequence may already be written down. Results marked stale have not been verified
    recently; treat their steps as a starting point, then re-store them with verified=true.
    """
    return await _safe(t.runbooks(_engine, PROJECT_ID, project_id=project_id, limit=limit))


@mcp.tool()
async def search(query: str, filters: dict | None = None) -> dict:
    """Search across ALL projects' knowledge. `filters` may include scope, limit, as_of.

    Use this (vs recall) for cross-project discovery. type/confidence filters are
    not yet applied and are reported back in `filters_ignored`.
    """
    return await _safe(t.search(_engine, query, filters=filters))


@mcp.tool()
async def relate(from_id: str, to_id: str, relationship_type: str) -> dict:
    """Manually link two knowledge entities (by their ids) with a typed relationship.

    Structural link only — shapes the graph but won't appear in semantic search.
    """
    return await _safe(t.relate(_engine, from_id, to_id, relationship_type))


@mcp.tool()
async def forget(knowledge_id: str, reason: str | None = None) -> dict:
    """Mark a fact (by id from recall/search) as no longer valid.

    Temporal end, NOT deletion — history is preserved (queryable via recall as_of).
    """
    return await _safe(t.forget(_engine, knowledge_id, reason=reason))


@mcp.tool()
async def update(knowledge_id: str, changes: dict) -> dict:
    """Supersede a fact with new content (`changes={"content": "..."}`).

    Creates a temporal version: the old fact is invalidated, the new one stored.
    """
    return await _safe(t.update(_engine, PROJECT_ID, knowledge_id, changes))


def main() -> None:
    logging.basicConfig(level=logging.INFO)  # stderr
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
