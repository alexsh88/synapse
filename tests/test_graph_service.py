"""Unit tests for GraphService (Phase 5) — fake driver, no live Neo4j."""

from __future__ import annotations

from datetime import datetime, timezone

from synapse.core.graph_queries import GraphService


class FakeResult:
    def __init__(self, records):
        self.records = records


class FakeDriver:
    """Dispatches canned records by matching a substring of the query."""

    def __init__(self, **canned):
        self.canned = canned

    async def execute_query(self, query, **params):
        q = query
        if "LIMIT 2000" in q:
            return FakeResult(self.canned.get("nodes", []))
        if "LIMIT 4000" in q:
            return FakeResult(self.canned.get("links", []))
        if "properties(n) AS props" in q:
            return FakeResult(self.canned.get("detail", []))
        if "AS target, r.name AS name, r.fact" in q:
            return FakeResult(self.canned.get("out", []))
        if "AS source, r.name AS name, r.fact" in q:
            return FakeResult(self.canned.get("inc", []))
        if "klabels" in q:
            return FakeResult(self.canned.get("timeline", []))
        if "STARTS WITH 'project_'" in q:
            return FakeResult(self.canned.get("projects", []))
        return FakeResult([])


class _FG:
    def __init__(self, driver):
        self.driver = driver


def svc(**canned):
    return GraphService(_FG(FakeDriver(**canned)))


class HealthFakeDriver:
    """Dispatches the four health() queries by a unique marker in each."""

    def __init__(self, counts, edges, promos, superseded, shared=0):
        self._counts, self._edges = counts, edges
        self._promos, self._superseded = promos, superseded
        self._shared = shared

    async def execute_query(self, query, **params):
        if "AS total_nodes" in query:
            return FakeResult([self._counts])
        if "AS active_edges" in query:
            return FakeResult([self._edges])
        if "AS shared" in query:
            return FakeResult([{"shared": self._shared}])
        if "promo_scopes" in query:
            return FakeResult(self._promos)
        if "ORDER BY r.invalid_at DESC" in query:
            return FakeResult(self._superseded)
        return FakeResult([])


async def test_snapshot_types_and_link_filtering():
    s = svc(
        nodes=[
            {"id": "a", "name": "A", "labels": ["Entity", "Decision"], "scope": "global", "summary": "sa", "degree": 2},
            {"id": "b", "name": "B", "labels": ["Entity"], "scope": "global", "summary": None, "degree": 1},
        ],
        links=[
            {"source": "a", "target": "b", "name": "AppliesTo", "fact": "A→B"},
            {"source": "a", "target": "zzz", "name": "X", "fact": "dangling"},
        ],
    )
    snap = await s.snapshot(["global"])
    assert {n.id for n in snap.nodes} == {"a", "b"}
    assert next(n for n in snap.nodes if n.id == "a").type == "decision"
    assert next(n for n in snap.nodes if n.id == "b").type == "entity"
    # the dangling link (target not in node set) is dropped
    assert len(snap.links) == 1 and snap.links[0].target == "b"


async def test_node_detail_strips_embeddings_and_returns_edges():
    s = svc(
        detail=[{"id": "a", "name": "A", "labels": ["Entity", "Lesson"], "scope": "project_x",
                 "summary": "s", "degree": 1,
                 "props": {"severity": "high", "name_embedding": [0.1], "uuid": "a", "name": "A"}}],
        out=[{"target": "b", "name": "DiscoveredIn", "fact": "A discovered in B"}],
        inc=[],
    )
    d = await s.node_detail("a")
    assert d.node.type == "lesson"
    assert d.attributes == {"severity": "high"}        # embedding + reserved keys stripped
    assert len(d.edges_out) == 1 and d.edges_out[0].target == "b"


async def test_node_detail_missing_returns_none():
    assert await svc(detail=[]).node_detail("zzz") is None


async def test_timeline_maps_type_and_native_datetime():
    when = datetime(2026, 6, 1, tzinfo=timezone.utc)
    s = svc(timeline=[{"id": "d1", "labels": ["Entity", "Decision"], "name": "Use X",
                       "scope": "project_acme-api", "created_at": when}])
    items = await s.timeline(["project_acme-api"])
    assert items[0].kind == "decision" and items[0].created_at == when


async def test_projects_rollup():
    s = svc(projects=[{"scope": "project_acme-api", "nodes": 10, "decisions": 3, "conventions": 5, "lessons": 2}])
    ps = await s.projects()
    assert ps[0].id == "acme-api" and ps[0].nodes == 10 and ps[0].decisions == 3


async def test_health_aggregates_counts_candidates_and_superseded():
    drv = HealthFakeDriver(
        counts={"total_nodes": 99, "decision": 10, "convention": 8, "lesson": 5,
                "research": 3, "pattern": 2, "tool": 1},
        edges={"total_edges": 50, "active_edges": 45, "superseded_edges": 5},
        shared=7,   # 7 concepts shared across >=2 projects (the real cross-project signal)
        # name shared across two projects → promotion candidate; label as Neo4j returns it
        promos=[{"name": "BigDecimal for money", "label": "Convention",
                 "promo_scopes": ["project_acme-api", "project_acme-data"]}],
        superseded=[{"fact": "old approach", "scope": "project_acme-api",
                     "invalid_at": datetime(2026, 5, 1, tzinfo=timezone.utc)}],
    )
    h = await GraphService(_FG(drv)).health()

    assert h.total_nodes == 99
    assert (h.active_edges, h.superseded_edges, h.cross_project_links) == (45, 5, 7)
    by_type = {t.type: t.count for t in h.by_type}
    assert by_type["decision"] == 10 and by_type["tool"] == 1
    # zero-count types are omitted; only the six knowledge labels appear
    assert "entity" not in by_type

    cand = h.promotion_candidates[0]
    assert cand.name == "BigDecimal for money" and cand.type == "convention"
    assert cand.projects == ["acme-api", "acme-data"]   # project_ prefix stripped

    assert h.recently_superseded[0].fact == "old approach"
    assert h.recently_superseded[0].scope == "project_acme-api"
