"""Search / recall / brief routes (Phase 5)."""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, Query

from synapse.api.deps import get_engine
from synapse.core.retrieval_engine import Brief, Recalled
from synapse.core.schema import Scope

router = APIRouter(tags=["search"])


@router.get("/search", response_model=list[Recalled])
async def search(
    q: str,
    scope: str | None = None,
    limit: int = Query(default=20, ge=1, le=200),
    engine=Depends(get_engine),
):
    # Shared parser (see Scope.parse_request). This used to prepend `project_` to anything that
    # wasn't already prefixed, so `cluster_trading` searched `project_cluster_trading` — a scope
    # that does not exist — and returned nothing rather than erroring.
    group_id = Scope.group_id_for_request(scope)
    return await engine.search(q, group_ids=[group_id] if group_id else None, limit=limit)


@router.get("/recall", response_model=list[Recalled])
async def recall(
    q: str,
    project: str | None = None,
    as_of: datetime | None = None,
    limit: int = Query(default=10, ge=1, le=200),
    feedback: bool = Query(
        default=False,
        description="Count these results as impressions (roadmap item 14). Set ONLY for a real "
                    "consumption — an agent recall or the UserPromptSubmit hook. Leaving it off for "
                    "eval runs and UI browsing is what keeps the signal from becoming "
                    "self-referential.",
    ),
    engine=Depends(get_engine),
):
    return await engine.recall(q, project_id=project, limit=limit, as_of=as_of, feedback=feedback)


@router.get("/brief/{project_id}", response_model=Brief)
async def brief(project_id: str, engine=Depends(get_engine)):
    return await engine.brief(project_id)
