"""Knowledge write routes — remember / update / forget (Phase 5).

Each successful write publishes a KnowledgeEvent so the UI updates in real time.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from synapse.api.deps import get_engine
from synapse.core.provenance import resolve as resolve_provenance
from synapse.core.schema import Scope
from synapse.api.events import KnowledgeEvent, bus
from synapse.models.api import ForgetResponse, RememberResponse, UpdateResponse

router = APIRouter(prefix="/knowledge", tags=["knowledge"])


class RememberBody(BaseModel):
    content: str
    # "global" / "project_x" / "x" / "cluster_trading" / "cluster:trading" / None(=global).
    # Parsed by Scope.parse_request — the same parser the MCP tools use, so the two interfaces
    # cannot drift. Cluster support was missing here until 2026-07-27: this route only stripped a
    # `project_` prefix, so `cluster:trading` was read as a PROJECT called "cluster:trading" and
    # Graphiti rejected the resulting `project_cluster:trading` group_id.
    type: str | None = None
    scope: str | None = None
    # Write attribution (roadmap item 13). Optional, and only worth sending when the caller knows
    # something the server cannot infer — a hook has the real session id, the API process does not.
    # Anything omitted falls back to the environment/host in synapse.core.provenance.resolve().
    agent: str | None = None
    model: str | None = None
    session_id: str | None = None


class UpdateBody(BaseModel):
    content: str


@router.post("", status_code=201, response_model=RememberResponse)
async def remember(body: RememberBody, engine=Depends(get_engine)):
    # No default project: an HTTP caller has no ambient project the way an MCP server does, so an
    # omitted scope means global (and the global-write gate may still refile it).
    cluster, project_id = Scope.parse_request(body.scope)
    result = await engine.remember(
        body.content, knowledge_type=body.type, project_id=project_id, cluster=cluster,
        provenance=resolve_provenance(
            agent=body.agent, model=body.model, session_id=body.session_id),
    )
    if result.outcome.value in ("stored", "contradiction"):
        await bus.publish(KnowledgeEvent(
            type="knowledge.added", id=result.episode_uuid, scope=result.scope,
            summary=(result.facts[0] if result.facts else None)))
    return result


@router.patch("/{knowledge_id}", response_model=UpdateResponse)
async def update(knowledge_id: str, body: UpdateBody, engine=Depends(get_engine)):
    result = await engine.update(knowledge_id, {"content": body.content})
    if result is None or (isinstance(result, dict) and not result.get("success", True) and result.get("not_found")):
        raise HTTPException(status_code=404, detail="knowledge not found")
    await bus.publish(KnowledgeEvent(type="knowledge.updated", id=knowledge_id))
    return result


@router.delete("/{knowledge_id}", response_model=ForgetResponse)
async def forget(knowledge_id: str, reason: str | None = None, engine=Depends(get_engine)):
    result = await engine.forget(knowledge_id, reason)
    if result is None or (isinstance(result, dict) and not result.get("success", True) and result.get("not_found")):
        raise HTTPException(status_code=404, detail="knowledge not found")
    await bus.publish(KnowledgeEvent(type="knowledge.forgotten", id=knowledge_id))
    return result
