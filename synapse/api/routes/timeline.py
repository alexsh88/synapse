"""Timeline route — chronological knowledge events (Phase 5)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from synapse.api.deps import get_engine
from synapse.core.graph_queries import TimelineItem

router = APIRouter(tags=["timeline"])


@router.get("/timeline", response_model=list[TimelineItem])
async def timeline(
    scope: list[str] = Query(default=["global"]),
    limit: int = Query(default=50, ge=1, le=200),
    engine=Depends(get_engine),
):
    return await engine.graph.timeline(scope, limit=limit)
