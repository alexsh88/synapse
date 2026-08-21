"""MCP tool implementations (plan Part 6).

Pure async functions over a ``KnowledgeEngine``. ``server.py`` wraps these as MCP
tools — injecting the connected engine and the project resolved from
``SYNAPSE_PROJECT_ID`` — and adds uniform error handling. Kept engine-agnostic so
they are unit-testable with a fake engine.

Every function returns a JSON-serializable ``dict``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from synapse.core.injection import REFERENCE_NOTICE
from synapse.core.schema import Scope


def _parse_as_of(as_of: str | None) -> datetime | None:
    if not as_of:
        return None
    return datetime.fromisoformat(as_of.replace("Z", "+00:00"))


# Thin wrappers over the one parser in Scope. These used to BE the parser, and the API routes had
# their own divergent versions that mangled a cluster scope — see Scope.parse_request.
def _scope_to_cluster(scope: str | None) -> str | None:
    """Extract a cluster name from ``cluster:trading`` / ``cluster_trading``, else None."""
    return Scope.parse_request(scope)[0]


def _scope_to_project(scope: str | None, default_project: str | None) -> str | None:
    """Resolve a tool ``scope`` arg to a project_id for the write/read paths.

    ``None`` -> the server's default project; ``"global"`` -> None (global scope);
    ``"project:x"`` / ``"project_x"`` / ``"x"`` -> that project id.

    A ``cluster:*`` scope yields None here, NOT the literal string: callers that support clusters
    read them from ``_scope_to_cluster`` first, and callers that don't (``recall`` composes the
    cluster tier from the project seat instead) previously turned it into the bogus project id
    ``cluster:trading``, which matched nothing and reported no error.
    """
    return Scope.parse_request(scope, default_project=default_project)[1]


async def remember(
    engine, default_project, content: str,
    type: str | None = None, scope: str | None = None, relationships: str | None = None,
) -> dict[str, Any]:
    cluster = _scope_to_cluster(scope)
    # A cluster write is domain-scoped, so it must not also carry a project scope.
    project_id = None if cluster else _scope_to_project(scope, default_project)
    if relationships:
        content = f"{content}\n\nKnown relationships: {relationships}"
    r = await engine.remember(content, knowledge_type=type, project_id=project_id,
                              cluster=cluster)
    return {
        "outcome": r.outcome.value, "knowledge_type": r.knowledge_type, "scope": r.scope,
        "episode_uuid": r.episode_uuid, "entities": r.entities, "facts": r.facts,
        "duplicate_of": r.duplicate_of, "contradicts": r.contradicts, "reason": r.reason,
        "degraded": r.degraded, "facts_extracted": r.facts_extracted,
        # Credential kinds stripped before storage (never the values). Surfaced so the agent
        # knows its content was altered and can stop echoing the secret.
        "redactions": r.redactions,
    }


async def recall(
    engine, default_project, query: str,
    scope: str | None = None, limit: int = 10, as_of: str | None = None,
) -> dict[str, Any]:
    cluster = _scope_to_cluster(scope)
    if cluster:
        # `recall` composes global+cluster+project FROM A PROJECT SEAT — it derives the cluster from
        # project_id and so has nowhere to put an explicitly requested one. Before the parsers were
        # unified this scope became the project id "cluster:trading", giving the group_id
        # `project_cluster:trading`, which matches nothing and reported no error. Falling through
        # with project_id=None would be no better: it would quietly recall from GLOBAL and drop the
        # caller's cluster entirely. So serve what was actually asked for, via the one path that can
        # target a single tier.
        hits = await engine.search(
            query, group_ids=[Scope.cluster(cluster)], limit=limit, as_of=_parse_as_of(as_of),
        )
    else:
        project_id = _scope_to_project(scope, default_project)
        # An agent asking for knowledge IS the consumption we want to measure (roadmap item 14).
        hits = await engine.recall(query, project_id=project_id, limit=limit,
                                   as_of=_parse_as_of(as_of), feedback=True)
    return {
        "_notice": REFERENCE_NOTICE,
        "query": query, "count": len(hits),
        "results": [
            {"fact": h.fact, "score": h.score, "scope": h.scope, "id": h.uuid,
             "valid_at": h.valid_at.isoformat() if h.valid_at else None}
            for h in hits
        ],
    }


async def remember_runbook(
    engine, default_project, name: str, steps: list[str],
    purpose: str | None = None, prerequisites: str | None = None,
    scope: str | None = None, verified: bool = True,
) -> dict[str, Any]:
    """Store an ordered procedure (roadmap item 18).

    Separate from ``remember`` on purpose. ``remember`` takes prose and lets extraction find the
    knowledge in it, which is exactly what flattens a procedure — the live graph held an
    `Acme-Jobs TDD workflow: failing test → implementation → integration → commit` node whose
    whole sequence had been squeezed into its name. ``steps`` is a list here so the ordering never
    has to survive a language model.

    ``verified`` stamps `verified_at` now, meaning "I just ran these and they worked". Pass false
    when recording a procedure you have not personally confirmed — an unverified runbook is
    reported as stale rather than presented as trustworthy.
    """
    if not steps:
        return {"error": "a runbook needs at least one step"}
    cluster = _scope_to_cluster(scope)
    project_id = None if cluster else _scope_to_project(scope, default_project)
    try:
        record = await engine.remember_runbook(
            name, list(steps), project_id=project_id, cluster=cluster,
            purpose=purpose, prerequisites=prerequisites,
            verified_at=datetime.now(timezone.utc) if verified else None,
        )
    except ValueError as exc:
        return {"error": str(exc)}
    return {
        "name": record.name, "scope": record.scope, "id": record.uuid,
        "steps": record.steps, "step_count": len(record.steps),
        "verified_at": record.verified_at.isoformat() if record.verified_at else None,
        # Non-empty means this REPLACED an existing procedure; the old steps are kept (R4).
        "superseded_steps": record.previous_steps,
    }


async def runbooks(
    engine, default_project, project_id: str | None = None, limit: int = 20,
) -> dict[str, Any]:
    """List procedures visible from a project's seat, steps intact."""
    records = await engine.runbooks(project_id or default_project, limit=limit)
    return {
        "count": len(records),
        "runbooks": [
            {"name": r.name, "scope": r.scope, "steps": r.steps, "purpose": r.purpose,
             "prerequisites": r.prerequisites, "stale": r.is_stale(),
             "verified_at": r.verified_at.isoformat() if r.verified_at else None}
            for r in records
        ],
    }


async def brief(engine, default_project, project_id: str | None = None) -> dict[str, Any]:
    pid = project_id or default_project
    if not pid:
        return {"error": "no project_id (set SYNAPSE_PROJECT_ID or pass project_id)"}
    b = await engine.brief(pid)
    # The brief is injected wholesale into every session's opening context by the hook, which
    # makes it the highest-reach path in the system and the one most worth labelling.
    return {**b.model_dump(mode="json"), "_notice": REFERENCE_NOTICE}


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
