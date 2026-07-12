"""Search / recall / brief routes (Phase 5)."""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, Query

from synapse.api.deps import get_engine
from synapse.core.retrieval_engine import Brief, Recalled

router = APIRouter(tags=["search"])


@router.get("/search", response_model=list[Recalled])
async def search(
    q: str,
    scope: str | None = None,
    limit: int = Query(default=20, ge=1, le=200),
    engine=Depends(get_engine),
):
    group_ids = None
    if scope:
        group_ids = ["global"] if scope == "global" else [
            scope if scope.startswith("project_") else f"project_{scope}"
        ]
    return await engine.search(q, group_ids=group_ids, limit=limit)


@router.get("/recall", response_model=list[Recalled])
async def recall(
    q: str,
    project: str | None = None,
    as_of: datetime | None = None,
    limit: int = Query(default=10, ge=1, le=200),
    engine=Depends(get_engine),
):
    return await engine.recall(q, project_id=project, limit=limit, as_of=as_of)


@router.get("/brief/{project_id}", response_model=Brief)
async def brief(project_id: str, engine=Depends(get_engine)):
    return await engine.brief(project_id)
