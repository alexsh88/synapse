"""BackupService — snapshot + zero-loss verification (R8). Fake driver, no live Neo4j."""

from __future__ import annotations

import pytest

from synapse.core.backup import BackupService, CurationSafetyError, GraphSnapshot


class FakeResult:
    def __init__(self, records):
        self.records = records


class FakeDriver:
    """Returns the same canned edges/nodes for every collect() (full or scoped)."""

    def __init__(self, edges, nodes):
        self._edges, self._nodes = edges, nodes
        self.scoped_edge_uuids: list[str] | None = None   # captured from a scoped edge query

    async def execute_query(self, query, **params):
        # Scoped one-hop node query returns endpoint uuids via `ep.uuid AS uuid`.
        if "ep.uuid AS uuid" in query:
            return FakeResult(self._nodes)
        if "merged_into AS merged_into" in query:
            if "$edge_uuids" in query:
                self.scoped_edge_uuids = params.get("edge_uuids")
            return FakeResult(self._edges)
        if "n.uuid AS uuid" in query:
            return FakeResult(self._nodes)
        return FakeResult([])


class NeighborhoodDriver:
    """A tiny concrete graph so a SCOPED collect can be checked to include exactly the one-hop neighborhood.

    Graph: (n1)-[e1]->(n2)-[e2]->(n3),  plus (n4)-[e3]->(n5) in a disjoint component.
    A scoped snapshot around e1 must include e1's endpoints {n1, n2}, and every edge incident to them
    ({e1, e2}) — but NOT e3 or its nodes.
    """

    _EDGES = {
        "e1": {"src": "n1", "dst": "n2"},
        "e2": {"src": "n2", "dst": "n3"},
        "e3": {"src": "n4", "dst": "n5"},
    }

    async def execute_query(self, query, **params):
        if "ep.uuid AS uuid" in query:  # scoped node query
            nodes = self._one_hop_nodes(params["edge_uuids"])
            return FakeResult([{"uuid": u} for u in sorted(nodes)])
        if "merged_into AS merged_into" in query:
            if "$edge_uuids" in query:  # scoped edge query: edges incident to the endpoint nodes
                nodes = self._one_hop_nodes(params["edge_uuids"])
                incident = [u for u, e in self._EDGES.items()
                            if e["src"] in nodes or e["dst"] in nodes]
                return FakeResult([_edge(u) for u in sorted(incident)])
            # full-graph edge query
            return FakeResult([_edge(u) for u in sorted(self._EDGES)])
        if "n.uuid AS uuid" in query:  # full-graph node query
            all_nodes = {n for e in self._EDGES.values() for n in (e["src"], e["dst"])}
            return FakeResult([{"uuid": u} for u in sorted(all_nodes)])
        return FakeResult([])

    def _one_hop_nodes(self, edge_uuids) -> set[str]:
        eps: set[str] = set()
        for u in edge_uuids:
            e = self._EDGES.get(u)
            if e:
                eps.update((e["src"], e["dst"]))
        return eps


class _FG:
    def __init__(self, driver):
        self.driver = driver


def _edge(uuid):
    return {"uuid": uuid, "fact": f"fact {uuid}", "group_id": "global",
            "valid_at": None, "invalid_at": None, "archived": None, "merged_into": None}


async def test_snapshot_writes_file_and_verify_passes(tmp_path):
    drv = FakeDriver(edges=[_edge("e1"), _edge("e2")], nodes=[{"uuid": "n1"}])
    svc = BackupService(_FG(drv), tmp_path)

    path = await svc.snapshot("test")
    assert path.exists() and path.suffix == ".json"

    res = await svc.verify_no_loss(path)              # current == backup → no loss
    assert res == {"ok": True, "edges_checked": 2, "nodes_checked": 1}


async def test_verify_raises_when_a_fact_vanished(tmp_path):
    # Current graph is missing e2 that the backup recorded → hard loss.
    drv = FakeDriver(edges=[_edge("e1")], nodes=[{"uuid": "n1"}])
    svc = BackupService(_FG(drv), tmp_path)
    before = GraphSnapshot(taken_at="2026-06-03T00:00:00+00:00",
                           edges=[_edge("e1"), _edge("e2")], node_uuids=["n1"])

    with pytest.raises(CurationSafetyError, match="e2"):
        await svc.verify_no_loss(before)


async def test_verify_raises_when_an_entity_node_vanished(tmp_path):
    drv = FakeDriver(edges=[_edge("e1")], nodes=[{"uuid": "n1"}])
    svc = BackupService(_FG(drv), tmp_path)
    before = GraphSnapshot(taken_at="2026-06-03T00:00:00+00:00",
                           edges=[_edge("e1")], node_uuids=["n1", "n2"])

    with pytest.raises(CurationSafetyError):
        await svc.verify_no_loss(before)


# --- scoped one-hop snapshots (WP-H item 2) -------------------------------------


async def test_scoped_collect_includes_exactly_one_hop_neighborhood(tmp_path):
    # Around e1 (n1->n2): endpoints {n1,n2}; incident edges {e1,e2}. e3/n4/n5 stay out.
    svc = BackupService(_FG(NeighborhoodDriver()), tmp_path)
    snap = await svc.collect(edge_uuids=["e1"])
    assert set(snap.edge_uuids) == {"e1", "e2"}
    assert set(snap.node_uuids) == {"n1", "n2"}


async def test_full_collect_unchanged_when_no_edge_uuids(tmp_path):
    svc = BackupService(_FG(NeighborhoodDriver()), tmp_path)
    snap = await svc.collect()
    assert set(snap.edge_uuids) == {"e1", "e2", "e3"}
    assert set(snap.node_uuids) == {"n1", "n2", "n3", "n4", "n5"}


async def test_scoped_snapshot_verify_passes_after_supersede(tmp_path):
    # A supersede/merge sets invalid_at/merged_into but the edge+nodes still EXIST → no loss.
    drv = FakeDriver(edges=[_edge("e1"), _edge("e2")], nodes=[{"uuid": "n1"}, {"uuid": "n2"}])
    svc = BackupService(_FG(drv), tmp_path)
    before = await svc.collect(edge_uuids=["e1"])          # scoped snapshot of the affected region
    assert drv.scoped_edge_uuids == ["e1"]                 # collect used the scoped query
    # Simulate a supersede: property mutation only, nothing removed.
    merged = _edge("e1"); merged["invalid_at"] = "2026-06-03T00:00:00Z"; merged["merged_into"] = "e2"
    drv._edges = [merged, _edge("e2")]
    res = await svc.verify_no_loss(before)                 # everything in the snapshot still resolves
    assert res == {"ok": True, "edges_checked": 2, "nodes_checked": 2}


async def test_scoped_snapshot_verify_catches_a_lost_edge(tmp_path):
    # A scoped snapshot must still CATCH a hard loss inside its neighborhood (regression guard, R8).
    drv = FakeDriver(edges=[_edge("e1"), _edge("e2")], nodes=[{"uuid": "n1"}, {"uuid": "n2"}])
    svc = BackupService(_FG(drv), tmp_path)
    before = await svc.collect(edge_uuids=["e1"])
    drv._edges = [_edge("e1")]                             # e2 was hard-deleted → loss
    with pytest.raises(CurationSafetyError, match="e2"):
        await svc.verify_no_loss(before)
