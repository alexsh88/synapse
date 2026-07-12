"""Knowledge write routes — remember / update / forget (Phase 5).

Each successful write publishes a KnowledgeEvent so the UI updates in real time.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from synapse.api.deps import get_engine
from synapse.api.events import KnowledgeEvent, bus
from synapse.models.api import ForgetResponse, RememberResponse, UpdateResponse

router = APIRouter(prefix="/knowledge", tags=["knowledge"])


def _project_from_scope(scope: str | None) -> str | None:
    if not scope or scope == "global":
        return None
    return scope.removeprefix("project_")


class RememberBody(BaseModel):
    content: str
    type: str | None = None
    scope: str | None = None       # "global" / "project_x" / "x" / None(=global)


class UpdateBody(BaseModel):
    content: str


@router.post("", status_code=201, response_model=RememberResponse)
async def remember(body: RememberBody, engine=Depends(get_engine)):
    result = await engine.remember(
        body.content, knowledge_type=body.type, project_id=_project_from_scope(body.scope))
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
