"""Graph routes — force-graph data + node detail (Phase 5)."""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query

from synapse.api.deps import get_engine
from synapse.core.graph_queries import GraphSnapshot, NodeDetail

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
