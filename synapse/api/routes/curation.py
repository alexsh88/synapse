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
from synapse.core.consolidation_engine import ConsolidationRun, Proposal, ProposalResult
from synapse.core.curation_engine import ApplyResult, CurationSuggestions
from synapse.core.graph_queries import CurationHealth

router = APIRouter(prefix="/curation", tags=["curation"])


class ApplyRequest(BaseModel):
    action: Literal["merge", "archive", "restore"]
    edge_uuid: str
    canonical_uuid: str | None = None  # required for merge


class ProposalApplyRequest(BaseModel):
    # Promotions are synthesis, not relocation — the reviewer supplies the statement that gets
    # stored at the wider scope. Merges ignore this field.
    statement: str | None = None


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


# --- consolidation review inbox (research Wave 2) ----------------------------
# The nightly worker only ever PROPOSES. These endpoints are the human review loop: list what it
# suggested, apply the ones you agree with, dismiss the rest (dismissed never comes back).


@router.get("/proposals", response_model=list[Proposal])
async def list_proposals(
    status: str | None = "open",
    kind: Literal["merge", "promote", "contradiction"] | None = None,
    limit: int = 100,
    engine=Depends(get_engine),
):
    return await engine.consolidation.list_proposals(status=status, kind=kind, limit=limit)


@router.post("/consolidate", response_model=ConsolidationRun)
async def run_consolidation(
    max_merges: int = 50, max_promotions: int = 25, max_contradictions: int = 20,
    engine=Depends(get_engine),
):
    """Generate proposals now instead of waiting for the nightly beat. Never mutates knowledge."""
    return await engine.consolidation.propose(
        max_merges=max_merges, max_promotions=max_promotions,
        max_contradictions=max_contradictions,
    )


@router.post("/proposals/{uuid}/apply", response_model=ProposalResult)
async def apply_proposal(
    uuid: str, req: ProposalApplyRequest | None = None, engine=Depends(get_engine),
):
    try:
        result = await engine.consolidation.apply(
            uuid, statement=(req.statement if req else None),
        )
    except CurationSafetyError as exc:
        raise HTTPException(status_code=409, detail=f"curation safety check failed: {exc}") from exc
    if not result.ok and result.needs_statement:
        # 422, not 400: the request was understood but is incomplete for this proposal kind.
        raise HTTPException(status_code=422, detail=result.detail)
    if not result.ok:
        raise HTTPException(status_code=404 if "not found" in result.detail else 409,
                            detail=result.detail)
    return result


@router.post("/proposals/{uuid}/dismiss", response_model=ProposalResult)
async def dismiss_proposal(uuid: str, engine=Depends(get_engine)):
    result = await engine.consolidation.dismiss(uuid)
    if not result.ok:
        raise HTTPException(status_code=404, detail=result.detail)
    return result
