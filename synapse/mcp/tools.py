"""MCP tool implementations (plan Part 6).

Pure async functions over a ``KnowledgeEngine``. ``server.py`` wraps these as MCP
tools — injecting the connected engine and the project resolved from
``SYNAPSE_PROJECT_ID`` — and adds uniform error handling. Kept engine-agnostic so
they are unit-testable with a fake engine.

Every function returns a JSON-serializable ``dict``.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any


def _parse_as_of(as_of: str | None) -> datetime | None:
    if not as_of:
        return None
    return datetime.fromisoformat(as_of.replace("Z", "+00:00"))


def _scope_to_project(scope: str | None, default_project: str | None) -> str | None:
    """Resolve a tool ``scope`` arg to a project_id for the write/read paths.

    ``None`` -> the server's default project; ``"global"`` -> None (global scope);
    ``"project:x"`` / ``"project_x"`` / ``"x"`` -> that project id.
    """
    if scope is None:
        return default_project
    s = scope.strip().lower()
    if s == "global":
        return None
    if s.startswith("project:"):
        return scope.split(":", 1)[1]
    if s.startswith("project_"):
        return scope[len("project_"):]
    return scope


async def remember(
    engine, default_project, content: str,
    type: str | None = None, scope: str | None = None, relationships: str | None = None,
) -> dict[str, Any]:
    project_id = _scope_to_project(scope, default_project)
    if relationships:
        content = f"{content}\n\nKnown relationships: {relationships}"
    r = await engine.remember(content, knowledge_type=type, project_id=project_id)
    return {
        "outcome": r.outcome.value, "knowledge_type": r.knowledge_type, "scope": r.scope,
        "episode_uuid": r.episode_uuid, "entities": r.entities, "facts": r.facts,
        "duplicate_of": r.duplicate_of, "contradicts": r.contradicts, "reason": r.reason,
        "degraded": r.degraded, "facts_extracted": r.facts_extracted,
    }


async def recall(
    engine, default_project, query: str,
    scope: str | None = None, limit: int = 10, as_of: str | None = None,
) -> dict[str, Any]:
    project_id = _scope_to_project(scope, default_project)
    hits = await engine.recall(query, project_id=project_id, limit=limit, as_of=_parse_as_of(as_of))
    return {
        "query": query, "count": len(hits),
        "results": [
            {"fact": h.fact, "score": h.score, "scope": h.scope, "id": h.uuid,
             "valid_at": h.valid_at.isoformat() if h.valid_at else None}
            for h in hits
        ],
    }


async def brief(engine, default_project, project_id: str | None = None) -> dict[str, Any]:
    pid = project_id or default_project
    if not pid:
        return {"error": "no project_id (set SYNAPSE_PROJECT_ID or pass project_id)"}
    b = await engine.brief(pid)
    return b.model_dump(mode="json")


async def relate(engine, from_id: str, to_id: str, relationship_type: str) -> dict[str, Any]:
    return await engine.relate(from_id, to_id, relationship_type)


async def search(engine, query: str, filters: dict | None = None) -> dict[str, Any]:
    filters = filters or {}
    limit = int(filters.get("limit", 10))
    as_of = _parse_as_of(filters.get("as_of"))
    scope = filters.get("scope")
    group_ids: list[str] | None = None
    if scope:
        if scope == "global":
            group_ids = ["global"]
        elif scope.startswith("project:"):
            group_ids = [f"project_{scope.split(':', 1)[1]}"]
        else:
            group_ids = [scope]
    hits = await engine.search(query, group_ids=group_ids, limit=limit, as_of=as_of)
    return {
        "query": query, "count": len(hits),
        "results": [{"fact": h.fact, "score": h.score, "scope": h.scope, "id": h.uuid} for h in hits],
        "filters_applied": {"scope": scope, "limit": limit, "as_of": filters.get("as_of")},
        # type/confidence filters are best-effort future work — surfaced, not silently dropped.
        "filters_ignored": [k for k in ("type", "confidence") if k in filters],
    }


async def forget(engine, knowledge_id: str, reason: str | None = None) -> dict[str, Any]:
    return await engine.forget(knowledge_id, reason)


async def update(engine, default_project, knowledge_id: str, changes: dict) -> dict[str, Any]:
    return await engine.update(knowledge_id, changes, project_id=default_project)
