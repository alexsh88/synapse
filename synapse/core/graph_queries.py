"""GraphService — read queries that shape Neo4j data for the UI (Phase 5).

Force-graph snapshots, node detail, timeline, and project rollups. Routes stay thin
by delegating here. Uses Graphiti's driver directly; no LLM/embedder involved.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

# Typed knowledge labels (for timeline / project counts).
_KNOWLEDGE_LABELS = ["Decision", "Convention", "Lesson", "Research", "Pattern", "Tool"]


class GraphNode(BaseModel):
    id: str
    name: str
    type: str            # lowercased label, e.g. "decision"; "entity" if generic
    scope: str
    degree: int = 0
    summary: str | None = None


class GraphLink(BaseModel):
    source: str
    target: str
    name: str | None = None
    fact: str | None = None


class GraphSnapshot(BaseModel):
    nodes: list[GraphNode] = Field(default_factory=list)
    links: list[GraphLink] = Field(default_factory=list)


class NodeDetail(BaseModel):
    node: GraphNode
    attributes: dict = Field(default_factory=dict)
    edges_out: list[GraphLink] = Field(default_factory=list)
    edges_in: list[GraphLink] = Field(default_factory=list)


class ProjectSummary(BaseModel):
    id: str
    name: str
    nodes: int = 0
    decisions: int = 0
    conventions: int = 0
    lessons: int = 0


class TimelineItem(BaseModel):
    id: str
    kind: str
    name: str
    scope: str
    created_at: datetime | None = None


class TypeCount(BaseModel):
    type: str
    count: int


class PromotionCandidate(BaseModel):
    """Knowledge whose name recurs across projects — a candidate to lift to ``global``."""

    name: str
    type: str
    projects: list[str]


class SupersededItem(BaseModel):
    """A fact whose edge carries ``invalid_at`` — the temporal model made visible."""

    fact: str
    scope: str
    invalid_at: datetime | None = None


class CurationHealth(BaseModel):
    total_nodes: int = 0
    active_edges: int = 0
    superseded_edges: int = 0
    cross_project_links: int = 0
    by_type: list[TypeCount] = Field(default_factory=list)
    promotion_candidates: list[PromotionCandidate] = Field(default_factory=list)
    recently_superseded: list[SupersededItem] = Field(default_factory=list)


def _type_of(labels: list[str]) -> str:
    for label in labels or []:
        if label != "Entity":
            return label.lower()
    return "entity"


def _native(value):
    return value.to_native() if hasattr(value, "to_native") else value


class GraphService:
    def __init__(self, graphiti) -> None:
        self._driver = graphiti.driver

    async def snapshot(
        self, scopes: list[str], types: list[str] | None = None,
        as_of: datetime | None = None, include_superseded: bool = False,
    ) -> GraphSnapshot:
        params: dict = {"scopes": scopes}
        type_filter = ""
        if types:
            params["types"] = [t.lower() for t in types]
            type_filter = "AND any(l IN labels(n) WHERE toLower(l) IN $types)"

        node_res = await self._driver.execute_query(
            f"""
            MATCH (n:Entity) WHERE n.group_id IN $scopes {type_filter}
            OPTIONAL MATCH (n)-[r:RELATES_TO]-()
            WITH n, count(r) AS degree
            RETURN n.uuid AS id, n.name AS name, labels(n) AS labels,
                   n.group_id AS scope, n.summary AS summary, degree
            ORDER BY degree DESC LIMIT 2000
            """,
            **params,
        )
        nodes = [
            GraphNode(id=r["id"], name=r["name"], type=_type_of(r["labels"]),
                      scope=r["scope"], summary=r["summary"], degree=int(r["degree"]))
            for r in node_res.records
        ]
        node_ids = {n.id for n in nodes}

        if as_of is not None:
            temporal = ("AND (r.valid_at IS NULL OR r.valid_at <= $as_of) "
                        "AND (r.invalid_at IS NULL OR r.invalid_at > $as_of)")
            params["as_of"] = as_of
        elif include_superseded:
            temporal = ""
        else:
            temporal = "AND r.invalid_at IS NULL"

        link_res = await self._driver.execute_query(
            f"""
            MATCH (n:Entity)-[r:RELATES_TO]->(m:Entity)
            WHERE n.group_id IN $scopes AND m.group_id IN $scopes
              AND coalesce(r.archived, false) = false {temporal}
            RETURN n.uuid AS source, m.uuid AS target, r.name AS name, r.fact AS fact
            LIMIT 4000
            """,
            **params,
        )
        links = [
            GraphLink(source=r["source"], target=r["target"], name=r["name"], fact=r["fact"])
            for r in link_res.records
            if r["source"] in node_ids and r["target"] in node_ids
        ]
        return GraphSnapshot(nodes=nodes, links=links)

    async def node_detail(self, uuid: str) -> NodeDetail | None:
        node_res = await self._driver.execute_query(
            """
            MATCH (n:Entity {uuid: $id})
            OPTIONAL MATCH (n)-[r:RELATES_TO]-()
            WITH n, count(r) AS degree
            RETURN n.uuid AS id, n.name AS name, labels(n) AS labels,
                   n.group_id AS scope, n.summary AS summary, degree, properties(n) AS props
            """,
            id=uuid,
        )
        if not node_res.records:
            return None
        r = node_res.records[0]
        props = {k: v for k, v in dict(r["props"] or {}).items()
                 if not k.endswith("_embedding") and k not in
                 ("uuid", "name", "summary", "group_id", "created_at", "labels")}
        node = GraphNode(id=r["id"], name=r["name"], type=_type_of(r["labels"]),
                         scope=r["scope"], summary=r["summary"], degree=int(r["degree"]))

        out_res = await self._driver.execute_query(
            "MATCH (n:Entity {uuid:$id})-[r:RELATES_TO]->(m:Entity) "
            "RETURN m.uuid AS target, r.name AS name, r.fact AS fact", id=uuid)
        in_res = await self._driver.execute_query(
            "MATCH (m:Entity)-[r:RELATES_TO]->(n:Entity {uuid:$id}) "
            "RETURN m.uuid AS source, r.name AS name, r.fact AS fact", id=uuid)
        return NodeDetail(
            node=node, attributes=props,
            edges_out=[GraphLink(source=uuid, target=r["target"], name=r["name"], fact=r["fact"]) for r in out_res.records],
            edges_in=[GraphLink(source=r["source"], target=uuid, name=r["name"], fact=r["fact"]) for r in in_res.records],
        )

    async def timeline(self, scopes: list[str], limit: int = 50) -> list[TimelineItem]:
        res = await self._driver.execute_query(
            """
            MATCH (n:Entity) WHERE n.group_id IN $scopes
              AND any(l IN labels(n) WHERE l IN $klabels) AND n.created_at IS NOT NULL
            RETURN n.uuid AS id, labels(n) AS labels, n.name AS name,
                   n.group_id AS scope, n.created_at AS created_at
            ORDER BY n.created_at DESC LIMIT $limit
            """,
            scopes=scopes, klabels=_KNOWLEDGE_LABELS, limit=limit,
        )
        return [TimelineItem(id=r["id"], kind=_type_of(r["labels"]), name=r["name"],
                             scope=r["scope"], created_at=_native(r["created_at"]))
                for r in res.records]

    async def projects(self) -> list[ProjectSummary]:
        res = await self._driver.execute_query(
            """
            MATCH (n:Entity) WHERE n.group_id STARTS WITH 'project_'
            RETURN n.group_id AS scope, count(n) AS nodes,
                   count(CASE WHEN n:Decision THEN 1 END) AS decisions,
                   count(CASE WHEN n:Convention THEN 1 END) AS conventions,
                   count(CASE WHEN n:Lesson THEN 1 END) AS lessons
            ORDER BY nodes DESC
            """,
        )
        out = []
        for r in res.records:
            pid = r["scope"].removeprefix("project_")
            out.append(ProjectSummary(id=pid, name=pid, nodes=int(r["nodes"]),
                                      decisions=int(r["decisions"]), conventions=int(r["conventions"]),
                                      lessons=int(r["lessons"])))
        return out

    async def health(self) -> CurationHealth:
        """Curation signals computable without the (Phase-10) curation engine:
        scale counts, the active/superseded split, cross-scope links, cross-project
        name collisions (promotion candidates), and recently superseded facts.
        """
        # Node total + per-type distribution (markers: "AS total_nodes").
        counts_res = await self._driver.execute_query(
            """
            MATCH (n:Entity)
            RETURN count(n) AS total_nodes,
                   count(CASE WHEN 'Decision'   IN labels(n) THEN 1 END) AS decision,
                   count(CASE WHEN 'Convention' IN labels(n) THEN 1 END) AS convention,
                   count(CASE WHEN 'Lesson'     IN labels(n) THEN 1 END) AS lesson,
                   count(CASE WHEN 'Research'   IN labels(n) THEN 1 END) AS research,
                   count(CASE WHEN 'Pattern'    IN labels(n) THEN 1 END) AS pattern,
                   count(CASE WHEN 'Tool'       IN labels(n) THEN 1 END) AS tool
            """,
        )
        c = counts_res.records[0] if counts_res.records else {}
        by_type = [TypeCount(type=label, count=int(c.get(label, 0) or 0))
                   for label in ("decision", "convention", "lesson", "research", "pattern", "tool")
                   if int(c.get(label, 0) or 0) > 0]

        # Edge split: active vs superseded, plus cross-scope links (marker: "AS active_edges").
        edge_res = await self._driver.execute_query(
            """
            MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity)
            RETURN count(CASE WHEN r.invalid_at IS NULL     THEN 1 END) AS active_edges,
                   count(CASE WHEN r.invalid_at IS NOT NULL THEN 1 END) AS superseded_edges
            """,
        )
        e = edge_res.records[0] if edge_res.records else {}

        # Cross-project = concepts shared across ≥2 project scopes. Graphiti partitions by group_id so
        # there are no cross-scope EDGES (the old count was structurally 0); the real signal is a node
        # name appearing in multiple projects (the same thing the promotion candidates surface).
        xproj_res = await self._driver.execute_query(
            """
            MATCH (n:Entity) WHERE n.group_id STARTS WITH 'project_'
            WITH n.name AS name, collect(DISTINCT n.group_id) AS scopes
            WHERE size(scopes) >= 2
            RETURN count(*) AS shared
            """,
        )
        cross_project = int(xproj_res.records[0]["shared"]) if xproj_res.records else 0

        # Names recurring across ≥2 project scopes → promotion candidates (marker: "promo_scopes").
        promo_res = await self._driver.execute_query(
            """
            MATCH (n:Entity) WHERE n.group_id STARTS WITH 'project_'
              AND any(l IN labels(n) WHERE l IN $klabels)
            WITH n.name AS name,
                 head([l IN labels(n) WHERE l <> 'Entity']) AS label,
                 collect(DISTINCT n.group_id) AS promo_scopes
            WHERE size(promo_scopes) >= 2
            RETURN name, label, promo_scopes
            ORDER BY size(promo_scopes) DESC, name LIMIT 20
            """,
            klabels=_KNOWLEDGE_LABELS,
        )
        promotion_candidates = [
            PromotionCandidate(
                name=r["name"], type=(r["label"] or "Entity").lower(),
                projects=[s.removeprefix("project_") for s in r["promo_scopes"]],
            )
            for r in promo_res.records
        ]

        # Recently superseded facts — the temporal model made visible (marker: "ORDER BY r.invalid_at DESC").
        sup_res = await self._driver.execute_query(
            """
            MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity) WHERE r.invalid_at IS NOT NULL
            RETURN r.fact AS fact, a.group_id AS scope, r.invalid_at AS invalid_at
            ORDER BY r.invalid_at DESC LIMIT 10
            """,
        )
        recently_superseded = [
            SupersededItem(fact=r["fact"] or "", scope=r["scope"] or "", invalid_at=_native(r["invalid_at"]))
            for r in sup_res.records
        ]

        return CurationHealth(
            total_nodes=int(c.get("total_nodes", 0) or 0),
            active_edges=int(e.get("active_edges", 0) or 0),
            superseded_edges=int(e.get("superseded_edges", 0) or 0),
            cross_project_links=cross_project,
            by_type=by_type,
            promotion_candidates=promotion_candidates,
            recently_superseded=recently_superseded,
        )
