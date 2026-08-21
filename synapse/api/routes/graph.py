"""Graph routes — force-graph data + node detail (Phase 5)."""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query

from synapse.api.deps import get_engine
from synapse.core.feedback import FeedbackSummary
from synapse.core.graph_queries import GraphSnapshot, NodeDetail, ProvenanceGroup

router = APIRouter(prefix="/graph", tags=["graph"])


@router.get("", response_model=GraphSnapshot)
async def get_graph(
    scope: list[str] = Query(default=["global"]),
    types: str | None = None,
    as_of: datetime | None = None,
    include_superseded: bool = False,
    engine=Depends(get_engine),
):
    type_list = [t.strip() for t in types.split(",") if t.strip()] if types else None
    return await engine.graph.snapshot(scope, types=type_list, as_of=as_of,
                                       include_superseded=include_superseded)


@router.get("/node/{node_id}", response_model=NodeDetail)
async def get_node(node_id: str, engine=Depends(get_engine)):
    detail = await engine.graph.node_detail(node_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="node not found")
    return detail


@router.get("/provenance", response_model=list[ProvenanceGroup])
async def provenance(
    session_id: str | None = None, agent: str | None = None, limit: int = 50,
    engine=Depends(get_engine),
):
    """Who taught Synapse what (roadmap item 13).

    Filter by ``session_id`` to see the blast radius of one session before deciding to roll it back.
    """
    return await engine.graph.provenance(session_id=session_id, agent=agent, limit=limit)


@router.get("/feedback", response_model=FeedbackSummary)
async def feedback(limit: int = 15, engine=Depends(get_engine)):
    """What retrieval actually delivers (roadmap item 14) — impressions, coverage, corrections.

    There is deliberately no "was it used" figure: that is not observable, and a fabricated signal
    wired into ranking would be worse than none.
    """
    return await engine.graph.feedback(limit=limit)
