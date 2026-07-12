"""API route + WebSocket tests (Phase 5) — fake engine via dependency override."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from synapse.api.deps import get_engine
from synapse.config import settings
from synapse.api.main import app
from synapse.core.curation_engine import (
    ApplyResult, CurationSuggestions, DuplicateCluster, FactRef,
)
from synapse.core.graph_queries import (
    CurationHealth, GraphNode, GraphSnapshot, NodeDetail, ProjectSummary,
    PromotionCandidate, SupersededItem, TimelineItem, TypeCount,
)
from synapse.core.retrieval_engine import Brief, Recalled
from synapse.core.write_pipeline import Outcome, WriteResult


class FakeGraph:
    async def snapshot(self, scopes, types=None, as_of=None, include_superseded=False):
        return GraphSnapshot(nodes=[GraphNode(id="a", name="A", type="decision", scope="global", degree=1)], links=[])

    async def node_detail(self, uuid):
        if uuid == "missing":
            return None
        return NodeDetail(node=GraphNode(id=uuid, name="A", type="decision", scope="global"))

    async def timeline(self, scopes, limit=50):
        return [TimelineItem(id="t1", kind="lesson", name="L", scope="global",
                             created_at=datetime(2026, 6, 1, tzinfo=timezone.utc))]

    async def projects(self):
        return [ProjectSummary(id="acme-api", name="acme-api", nodes=5, decisions=2)]

    async def health(self):
        return CurationHealth(
            total_nodes=99, active_edges=45, superseded_edges=5, cross_project_links=7,
            by_type=[TypeCount(type="decision", count=10)],
            promotion_candidates=[PromotionCandidate(
                name="BigDecimal", type="convention", projects=["acme-api", "acme-data"])],
            recently_superseded=[SupersededItem(fact="old", scope="global")],
        )


class FakeCuration:
    def __init__(self):
        self.calls: list = []

    async def suggestions(self):
        return CurationSuggestions(
            duplicates=[DuplicateCluster(
                scope="project_acme-api",
                canonical=FactRef(uuid="e1", fact="canonical fact"),
                duplicates=[FactRef(uuid="e2", fact="dup fact")],
                max_similarity=0.95,
            )],
        )

    async def merge_duplicate(self, canonical, dup):
        self.calls.append(("merge", canonical, dup))
        return ApplyResult(ok=True, action="merge", edge_uuid=dup, backup_path="backups/merge-x.json")

    async def archive(self, edge_uuid):
        self.calls.append(("archive", edge_uuid))
        return ApplyResult(ok=True, action="archive", edge_uuid=edge_uuid, backup_path="backups/archive-x.json")

    async def restore(self, edge_uuid):
        self.calls.append(("restore", edge_uuid))
        return ApplyResult(ok=True, action="restore", edge_uuid=edge_uuid)


class FakeCapture:
    def __init__(self):
        self.calls: list = []

    async def capture(self, project_id, session_id, transcript):
        self.calls.append(("capture", project_id, session_id))
        return {"stored": ["a durable lesson"], "pending": ["a borderline one"], "skipped": False}

    async def list_pending(self, project=None):
        return [{"uuid": "pc1", "project_id": "acme-api", "content": "borderline lesson",
                 "type": "lesson", "confidence": 0.6, "reason": "maybe useful"}]

    async def count(self):
        return 3

    async def approve(self, uuid):
        self.calls.append(("approve", uuid))
        return {"ok": uuid == "pc1", "stored": "borderline lesson"} if uuid == "pc1" else {"ok": False, "error": "not found"}

    async def dismiss(self, uuid):
        self.calls.append(("dismiss", uuid))
        return {"ok": True}


class FakeEngine:
    def __init__(self):
        self.graph = FakeGraph()
        self.curation = FakeCuration()
        self.capture = FakeCapture()
        self.calls: list = []

        self._missing_ids: set[str] = set()

    async def remember(self, content, *, knowledge_type=None, project_id=None, force=False):
        self.calls.append(("remember", content, project_id))
        return WriteResult(outcome=Outcome.STORED, knowledge_type="decision",
                           scope="global" if project_id is None else f"project_{project_id}",
                           episode_uuid="ep1", facts=["a fact"])

    async def update(self, kid, changes):
        self.calls.append(("update", kid, changes))
        if kid in self._missing_ids:
            return {"success": False, "not_found": True}
        return {"success": True}

    async def forget(self, kid, reason=None):
        self.calls.append(("forget", kid, reason))
        if kid in self._missing_ids:
            return {"success": False, "not_found": True}
        return {"success": True}

    async def search(self, q, *, group_ids=None, limit=10):
        self.calls.append(("search", q, group_ids))
        return [Recalled(fact="sf", score=0.9, scope="global", uuid="u1")]

    async def recall(self, q, *, project_id=None, limit=10, as_of=None):
        self.calls.append(("recall", q, project_id))
        return [Recalled(fact="rf", score=0.8, scope="global", uuid="u2")]

    async def brief(self, pid):
        return Brief(project_id=pid, project_summary="s", active_conventions=[], key_decisions=[],
                     relevant_lessons=[], cross_project_knowledge=[],
                     generated_at=datetime(2026, 6, 2, tzinfo=timezone.utc))


@pytest.fixture
def client_engine():
    engine = FakeEngine()
    app.dependency_overrides[get_engine] = lambda: engine
    yield TestClient(app), engine
    app.dependency_overrides.clear()


def test_health(client_engine):
    client, _ = client_engine
    assert client.get("/api/v1/health").json() == {"status": "ok"}


def test_graph_snapshot(client_engine):
    client, _ = client_engine
    body = client.get("/api/v1/graph?scope=global").json()
    assert len(body["nodes"]) == 1 and body["nodes"][0]["type"] == "decision"


def test_node_detail_and_404(client_engine):
    client, _ = client_engine
    assert client.get("/api/v1/graph/node/a").status_code == 200
    assert client.get("/api/v1/graph/node/missing").status_code == 404


def test_projects(client_engine, monkeypatch):
    from synapse.core import registry

    monkeypatch.setattr(
        registry, "PROJECTS",
        {"acme-api": {"name": "Acme-API", "path": "/tmp/acme-api", "cluster": "test", "desc": "t"}},
    )
    client, _ = client_engine
    assert client.get("/api/v1/projects").json()[0]["id"] == "acme-api"


def test_projects_status_list(client_engine, monkeypatch):
    # Pin the registry: the route unions registry + graph, and the real registry
    # is a machine-local projects.json — tests must not depend on its contents.
    from synapse.core import registry

    pinned = {
        pid: {"name": pid.title(), "path": f"/tmp/{pid}", "cluster": "test", "desc": pid}
        for pid in ("acme-api", "acme-jobs", "acme-flow")
    }
    monkeypatch.setattr(registry, "PROJECTS", pinned)

    client, _ = client_engine
    body = client.get("/api/v1/projects").json()
    # pinned registry (3) + any graph-only project from the fake engine (union)
    assert isinstance(body, list) and len(body) >= 3
    ids = {p["id"] for p in body}
    assert {"acme-api", "acme-jobs", "acme-flow"} <= ids
    oly = next(p for p in body if p["id"] == "acme-api")
    assert oly["nodes"] == 5 and "connected" in oly and "hook" in oly


def test_connect_requires_path_for_unregistered(client_engine):
    client, _ = client_engine
    r = client.post("/api/v1/projects/connect", json={"id": "brand-new-xyz", "deep_seed": False})
    assert r.status_code == 422


def test_connect_status_unknown_job_404(client_engine):
    client, _ = client_engine
    assert client.get("/api/v1/projects/connect/nope").status_code == 404


def test_timeline(client_engine):
    client, _ = client_engine
    assert client.get("/api/v1/timeline?scope=global").json()[0]["kind"] == "lesson"


def test_remember_routes_scope(client_engine):
    client, engine = client_engine
    r = client.post("/api/v1/knowledge", json={"content": "we chose X", "scope": "project_acme-api"})
    assert r.json()["outcome"] == "stored" and r.json()["scope"] == "project_acme-api"
    assert engine.calls[0] == ("remember", "we chose X", "acme-api")


def test_update_and_forget(client_engine):
    client, engine = client_engine
    assert client.patch("/api/v1/knowledge/k1", json={"content": "new"}).json()["success"]
    assert client.delete("/api/v1/knowledge/k1?reason=stale").json()["success"]
    assert ("update", "k1", {"content": "new"}) in engine.calls
    assert ("forget", "k1", "stale") in engine.calls


def test_search_recall_brief(client_engine):
    client, _ = client_engine
    assert client.get("/api/v1/search?q=money").json()[0]["fact"] == "sf"
    assert client.get("/api/v1/recall?q=money&project=acme-api").json()[0]["fact"] == "rf"
    assert client.get("/api/v1/brief/acme-api").json()["project_id"] == "acme-api"


def test_curation_health(client_engine):
    client, _ = client_engine
    body = client.get("/api/v1/curation/health").json()
    assert body["total_nodes"] == 99 and body["cross_project_links"] == 7
    assert body["by_type"][0]["type"] == "decision"
    assert body["promotion_candidates"][0]["projects"] == ["acme-api", "acme-data"]


def test_curation_suggestions(client_engine):
    client, _ = client_engine
    body = client.get("/api/v1/curation/suggestions").json()
    assert body["duplicates"][0]["canonical"]["uuid"] == "e1"
    assert body["duplicates"][0]["duplicates"][0]["uuid"] == "e2"


def test_curation_apply_merge_archive_restore(client_engine):
    client, engine = client_engine
    m = client.post("/api/v1/curation/apply",
                    json={"action": "merge", "edge_uuid": "e2", "canonical_uuid": "e1"}).json()
    assert m["ok"] and m["action"] == "merge" and m["backup_path"]
    assert client.post("/api/v1/curation/apply", json={"action": "archive", "edge_uuid": "e2"}).json()["ok"]
    assert client.post("/api/v1/curation/apply", json={"action": "restore", "edge_uuid": "e2"}).json()["ok"]
    assert ("merge", "e1", "e2") in engine.curation.calls
    # merge without canonical_uuid → 422
    assert client.post("/api/v1/curation/apply", json={"action": "merge", "edge_uuid": "e2"}).status_code == 422


def test_capture_trigger_accepts_and_backgrounds(client_engine):
    client, _ = client_engine
    # non-trivial transcript is accepted (processed in the background)
    assert client.post("/api/v1/capture",
                       json={"project_id": "acme-api", "session_id": "s1", "transcript": "x" * 300}
                       ).json()["accepted"] is True
    # trivially short transcript is skipped synchronously
    assert client.post("/api/v1/capture",
                       json={"project_id": "acme-api", "transcript": "short"}).json()["accepted"] is False


def test_captures_list_count_and_review(client_engine):
    client, engine = client_engine
    assert client.get("/api/v1/captures").json()[0]["uuid"] == "pc1"
    assert client.get("/api/v1/captures/count").json()["count"] == 3
    assert client.post("/api/v1/captures/pc1/approve").json()["ok"] is True
    assert client.post("/api/v1/captures/nope/approve").status_code == 404
    assert client.post("/api/v1/captures/pc1/dismiss").json()["ok"] is True


def test_websocket_receives_write_event(client_engine):
    client, _ = client_engine
    with client.websocket_connect("/ws") as ws:
        assert ws.receive_json()["type"] == "hello"
        client.post("/api/v1/knowledge", json={"content": "a new decision", "scope": "global"})
        msg = ws.receive_json()
        assert msg["type"] == "knowledge.added" and msg["scope"] == "global"


# ── WP-D new tests ─────────────────────────────────────────────────────────────

def test_bad_as_of_returns_422(client_engine):
    """Spec item 2: invalid as_of date string => 422 (not 500)."""
    client, _ = client_engine
    r = client.get("/api/v1/graph?as_of=not-a-date")
    assert r.status_code == 422, f"graph got {r.status_code}: {r.text}"
    r = client.get("/api/v1/recall?q=x&as_of=not-a-date")
    assert r.status_code == 422, f"recall got {r.status_code}: {r.text}"


def test_limit_over_max_returns_422(client_engine):
    """Spec item 3: limit > 200 => 422."""
    client, _ = client_engine
    assert client.get("/api/v1/search?q=x&limit=100000").status_code == 422
    assert client.get("/api/v1/recall?q=x&limit=100000").status_code == 422
    assert client.get("/api/v1/timeline?limit=100000").status_code == 422
    assert client.get("/api/v1/captures?limit=100000").status_code == 422


def test_update_missing_id_returns_404(client_engine):
    """Spec item 4: update non-existent id => 404."""
    client, engine = client_engine
    engine._missing_ids.add("ghost")
    r = client.patch("/api/v1/knowledge/ghost", json={"content": "new"})
    assert r.status_code == 404, f"got {r.status_code}: {r.text}"


def test_forget_missing_id_returns_404(client_engine):
    """Spec item 4: forget non-existent id => 404."""
    client, engine = client_engine
    engine._missing_ids.add("ghost")
    r = client.delete("/api/v1/knowledge/ghost")
    assert r.status_code == 404, f"got {r.status_code}: {r.text}"


def test_remember_returns_201(client_engine):
    """Spec item 3: POST /knowledge => 201."""
    client, _ = client_engine
    r = client.post("/api/v1/knowledge", json={"content": "a new fact", "scope": "global"})
    assert r.status_code == 201, f"got {r.status_code}: {r.text}"


def test_auth_401_when_key_configured_and_missing(client_engine, monkeypatch):
    """Spec item 7: 401 when api_key configured and header absent."""
    client, _ = client_engine
    monkeypatch.setattr(settings, "api_key", "secret123")
    r = client.get("/api/v1/graph")
    assert r.status_code == 401, f"got {r.status_code}: {r.text}"


def test_auth_200_when_key_matches(client_engine, monkeypatch):
    """Spec item 7: 200 when X-Synapse-Key matches configured key."""
    client, _ = client_engine
    monkeypatch.setattr(settings, "api_key", "secret123")
    r = client.get("/api/v1/graph", headers={"X-Synapse-Key": "secret123"})
    assert r.status_code == 200, f"got {r.status_code}: {r.text}"


def test_auth_no_auth_when_key_empty(client_engine, monkeypatch):
    """Spec item 7: no auth required when api_key is empty."""
    client, _ = client_engine
    monkeypatch.setattr(settings, "api_key", "")
    r = client.get("/api/v1/graph")
    assert r.status_code == 200, f"got {r.status_code}: {r.text}"


def test_eventbus_drop_on_full():
    """Spec item 5: EventBus drops events (not blocking) when queue is full."""
    import asyncio
    import unittest.mock as mock
    from synapse.api.events import EventBus, KnowledgeEvent

    bus = EventBus()
    q = bus.subscribe()
    event = KnowledgeEvent(type="test.event")

    async def fill_and_overflow():
        # Fill to maxsize (100)
        for _ in range(100):
            await bus.publish(event)
        # 101st publish should NOT raise or block; it logs a warning and drops
        with mock.patch("synapse.api.events.logger") as mock_logger:
            await bus.publish(event)
            assert mock_logger.warning.called, "Expected warning log on queue full"

    asyncio.run(fill_and_overflow())
    assert q.qsize() == 100
    bus.unsubscribe(q)


# ── WP-D spec item 1+4: response_model wiring tests ───────────────────────────

def test_remember_response_includes_degraded_and_facts_extracted(client_engine):
    """Spec item 1+4: POST /knowledge response_model must expose degraded and facts_extracted."""
    client, _ = client_engine
    r = client.post("/api/v1/knowledge", json={"content": "a new decision", "scope": "global"})
    assert r.status_code == 201
    body = r.json()
    # Core fields
    assert body["outcome"] == "stored"
    assert body["scope"] == "global"
    # WP-B diagnostic fields — must NOT be stripped by response_model serialization
    assert "degraded" in body, f"degraded missing from response: {body}"
    assert "facts_extracted" in body, f"facts_extracted missing from response: {body}"
    assert body["degraded"] is False       # FakeEngine returns WriteResult with defaults
    assert body["facts_extracted"] == 0


def test_graph_snapshot_serializes_via_response_model(client_engine):
    """Spec item 2: GET /graph response correctly serializes through GraphSnapshot model."""
    client, _ = client_engine
    body = client.get("/api/v1/graph?scope=global").json()
    assert "nodes" in body and "links" in body
    node = body["nodes"][0]
    # GraphNode fields all present
    assert node["id"] == "a"
    assert node["type"] == "decision"
    assert node["scope"] == "global"
    assert "degree" in node


def test_search_serializes_recalled_list(client_engine):
    """Spec item 2: GET /search serializes through list[Recalled] response_model."""
    client, _ = client_engine
    body = client.get("/api/v1/search?q=money").json()
    assert isinstance(body, list) and len(body) == 1
    r = body[0]
    assert r["fact"] == "sf"
    assert r["score"] == pytest.approx(0.9)
    assert r["uuid"] == "u1"
    assert r["scope"] == "global"


def test_curation_health_serializes_via_response_model(client_engine):
    """Spec item 2: GET /curation/health serializes through CurationHealth response_model."""
    client, _ = client_engine
    body = client.get("/api/v1/curation/health").json()
    assert body["total_nodes"] == 99
    assert "by_type" in body
    assert "promotion_candidates" in body
    assert "recently_superseded" in body


def test_capture_accepted_serializes_via_response_model(client_engine):
    """Spec item 2: POST /capture serializes through CaptureAccepted response_model."""
    client, _ = client_engine
    r = client.post("/api/v1/capture",
                    json={"project_id": "acme-api", "session_id": "s1", "transcript": "x" * 300})
    assert r.status_code == 200
    body = r.json()
    assert "accepted" in body
    assert body["accepted"] is True


def test_queue_maxsize_single_source_of_truth():
    """Spec item 3: QUEUE_MAXSIZE re-exported from events.py is the same value."""
    from synapse.api.events import _QUEUE_MAXSIZE
    from synapse.models.api import QUEUE_MAXSIZE
    assert QUEUE_MAXSIZE == _QUEUE_MAXSIZE == 100
