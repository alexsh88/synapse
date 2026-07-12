"""Curation routes (Phase 9/10) — health signals + suggestions + human-approved apply.

Health/suggestions are read-only aggregates from the graph. ``apply`` performs a
**reversible, backup-first** mutation (merge / archive / restore) — the only path to a
destructive operation, always explicit. See ``docs/architecture/curation.md`` (R8).
"""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from synapse.api.deps import get_engine
from synapse.core.backup import CurationSafetyError
from synapse.core.curation_engine import ApplyResult, CurationSuggestions
from synapse.core.graph_queries import CurationHealth

router = APIRouter(prefix="/curation", tags=["curation"])


class ApplyRequest(BaseModel):
    action: Literal["merge", "archive", "restore"]
    edge_uuid: str
    canonical_uuid: str | None = None  # required for merge


@router.get("/health", response_model=CurationHealth)
async def curation_health(engine=Depends(get_engine)):
    return await engine.graph.health()


@router.get("/suggestions", response_model=CurationSuggestions)
async def curation_suggestions(engine=Depends(get_engine)):
    return await engine.curation.suggestions()


@router.post("/apply", response_model=ApplyResult)
async def curation_apply(req: ApplyRequest, engine=Depends(get_engine)):
    try:
        if req.action == "merge":
            if not req.canonical_uuid:
                raise HTTPException(status_code=422, detail="merge requires canonical_uuid")
            return await engine.curation.merge_duplicate(req.canonical_uuid, req.edge_uuid)
        if req.action == "archive":
            return await engine.curation.archive(req.edge_uuid)
        return await engine.curation.restore(req.edge_uuid)
    except CurationSafetyError as exc:
        # The zero-loss check failed post-mutation; the backup taken before the op is the recovery path.
        raise HTTPException(status_code=409, detail=f"curation safety check failed: {exc}") from exc
