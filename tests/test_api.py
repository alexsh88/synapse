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


class FakeReader:
    def __init__(self):
        self.invalidated: list[str] = []

    async def invalidate_brief(self, project_id):
        self.invalidated.append(project_id)


class FakeEngine:
    def __init__(self):
        self.graph = FakeGraph()
        self.curation = FakeCuration()
        self.capture = FakeCapture()
        self.reader = FakeReader()
        self.calls: list = []
        self.fail_remember = False

        self._missing_ids: set[str] = set()

    async def remember(self, content, *, knowledge_type=None, project_id=None, force=False,
                       provenance=None, **kw):
        self.calls.append(("remember", content, project_id))
        self.last_provenance = provenance
        if self.fail_remember:
            raise RuntimeError("extraction boom")
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

    async def recall(self, q, *, project_id=None, limit=10, as_of=None,
                     feedback=False, **kw):
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


# --- the connect SUCCESS path (2026-07-30) ------------------------------------------------
#
# Only the two rejection paths above were covered, so an endpoint that could never return 2xx
# shipped unnoticed. A real connect of a new project failed with a browser-visible 504 and left
# these two faults in the log:
#
#   1. `entity` was typed `dict | None` but holds the write OUTCOME string ("stored"), so every
#      successful connect died in response validation. The UI has always rendered it as text
#      (`Project entity: {job.entity}`) and the TS type has always said `entity?: string`, so the
#      string is the contract and the pydantic model was the odd one out.
#   2. The Project-entity write ran INLINE in the request. It is an LLM extraction: measured at
#      73s (wiring files written 09:45:08, handler reached its return at 09:46:22), which blew
#      nginx's default 60s proxy_read_timeout. The browser got a 504 while the work went on to
#      succeed in the background — the worst possible outcome, a false failure.


@pytest.fixture
def connectable(monkeypatch, tmp_path):
    """A real project folder under a throwaway projects_root.

    projects_root drives BOTH folder resolution and the connected-projects overlay that
    `add_connected` persists to, so pointing it at tmp keeps this test off the real registry.
    """
    from synapse.core import registry
    monkeypatch.setattr(settings, "projects_root", str(tmp_path))
    monkeypatch.setattr(registry, "PROJECTS", {})
    folder = tmp_path / "demo-project"
    folder.mkdir()
    return str(folder)


def _settled(client, job: dict, tries: int = 50) -> dict:
    """Poll until the background seed finishes — TestClient drives it on the same event loop."""
    for _ in range(tries):
        if job["state"] != "running":
            return job
        job = client.get(f"/api/v1/projects/connect/{job['job_id']}").json()
    raise AssertionError(f"job never settled: {job}")


def test_connect_returns_a_job_that_matches_its_response_model(client_engine, connectable):
    client, engine = client_engine
    r = client.post("/api/v1/projects/connect",
                    json={"id": "demo-project", "path": connectable, "deep_seed": False})
    assert r.status_code == 200, r.text          # was 500: entity typed dict, given "stored"
    job = _settled(client, r.json())
    assert job["state"] == "done" and job["entity"] == "stored"
    assert job["actions"], "the wiring actions belong in the response"
    assert engine.reader.invalidated == ["demo-project"], "a new project's brief must be dropped"


def test_connect_answers_before_the_seed_write_runs(client_engine, connectable):
    """Proven by making the write FAIL: inline it would 5xx the request; deferred, the request
    succeeds at filesystem speed and the JOB carries the failure."""
    client, engine = client_engine
    engine.fail_remember = True
    r = client.post("/api/v1/projects/connect",
                    json={"id": "demo-project", "path": connectable, "deep_seed": False})
    assert r.status_code == 200, r.text
    assert r.json()["state"] == "running", "no graph write may happen inside the request"
    job = _settled(client, r.json())
    assert job["state"] == "error" and "boom" in job["error"]


def test_a_deep_seed_still_reports_its_chunk_progress(client_engine, connectable, tmp_path):
    (tmp_path / "demo-project" / "README.md").write_text(
        "\n\n".join(f"Paragraph {i} of the project readme, long enough to clear the 80-character "
                    f"floor that the chunker applies to skip noise." for i in range(3)),
        encoding="utf-8")
    client, engine = client_engine
    r = client.post("/api/v1/projects/connect",
                    json={"id": "demo-project", "path": connectable, "deep_seed": True})
    assert r.status_code == 200, r.text
    job = _settled(client, r.json())
    assert job["state"] == "done" and job["entity"] == "stored"

    seeded = [content for kind, content, _ in engine.calls if kind == "remember"]
    assert sum("of the project readme" in c for c in seeded) == 3
    assert job["total"] == len(seeded) - 1 and job["stored"] == job["total"]  # -1: the entity write
    # Exactly the README's 3 — write_files also creates a CLAUDE.md, but it holds only Synapse's
    # own block and _chunks now strips that rather than seeding our boilerplate as project
    # knowledge. Guards the loop where the connector feeds its own output back in.
    assert job["total"] == 3


# --- connecting a project that lives outside the primary root (2026-07-31) ----------------
#
# Reported as `project folder not found: /projects/acme-mobile` for a folder that
# existed at C:\Users\dev\acme-mobile. Resolution keeps only the folder NAME — the
# container reaches host directories through bind mounts, so a host path must be re-rooted at its
# mount point — but it re-rooted that name at ONE hardcoded root. Every project outside it was
# unconnectable, and the 404 quoted a container path the user had never typed.


@pytest.fixture
def out_of_root(monkeypatch, tmp_path):
    """A project reachable only through a SECOND root, not a subdirectory of the first.

    That is the real shape of it: the container sees each out-of-root project through its own
    bind mount, so the alternative location is a root in its own right.
    """
    from synapse.core import registry
    primary, extra = tmp_path / "primary", tmp_path / "extra"
    primary.mkdir()
    folder = extra / "story-app"
    folder.mkdir(parents=True)
    monkeypatch.setattr(settings, "projects_root", str(primary))
    monkeypatch.setattr(settings, "extra_project_roots", str(extra))
    monkeypatch.setattr(registry, "PROJECTS", {})
    return folder


def test_connect_finds_a_project_under_an_extra_root(client_engine, out_of_root):
    client, _ = client_engine
    r = client.post("/api/v1/projects/connect",
                    json={"id": "story-app", "path": r"C:\Users\dev\story-app",
                          "deep_seed": False})
    assert r.status_code == 200, r.text
    assert _settled(client, r.json())["state"] == "done"
    assert (out_of_root / "CLAUDE.md").exists(), "wiring belongs in the real folder, not a guess"


def test_a_project_that_exists_under_no_root_says_which_roots_it_searched(client_engine,
                                                                         out_of_root):
    """The old message named one container path and called it "not found", which sent the reader
    looking for a missing folder instead of a missing ROOT."""
    client, _ = client_engine
    r = client.post("/api/v1/projects/connect",
                    json={"id": "nowhere-app", "path": "/wherever/nowhere-app",
                          "deep_seed": False})
    assert r.status_code == 404
    detail = r.json()["detail"]
    assert "nowhere-app" in detail
    assert str(out_of_root.parent) in detail, "every root actually searched must be named"


def test_a_traversing_path_is_still_rejected(client_engine, out_of_root):
    """Extra roots widen where a project may live; they do not weaken the traversal guard."""
    client, _ = client_engine
    r = client.post("/api/v1/projects/connect",
                    json={"id": "escape", "path": "..", "deep_seed": False})
    assert r.status_code == 400


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


# --- write provenance (roadmap item 13) ---------------------------------------


def test_remember_attributes_the_write_from_the_request_body(client_engine):
    client, engine = client_engine
    r = client.post("/api/v1/knowledge", json={
        "content": "we chose X", "scope": "acme-store",
        "agent": "claude-code", "model": "claude-opus-5", "session_id": "sess-42",
    })
    assert r.status_code == 201
    prov = engine.last_provenance
    assert prov is not None
    assert prov.agent == "claude-code" and prov.model == "claude-opus-5"
    assert prov.session_id == "sess-42"


def test_remember_still_attributes_when_the_body_omits_provenance(client_engine):
    # A caller that sends nothing must not produce an anonymous write — the host is always known.
    client, engine = client_engine
    r = client.post("/api/v1/knowledge", json={"content": "we chose Y", "scope": "acme-store"})
    assert r.status_code == 201
    assert engine.last_provenance is not None
    assert not engine.last_provenance.is_empty()


def test_remember_response_forwards_the_pipelines_diagnostic_fields(client_engine):
    # extra="allow" does NOT forward fields off a returned object — pydantic only harvests extras
    # from a mapping. Every field a caller needs must be declared on RememberResponse. Found live:
    # the global-write gate refiled a write but scope_redirected_from and reason both came back null.
    from synapse.models.api import RememberResponse

    declared = set(RememberResponse.model_fields)
    for field in ("reason", "redactions", "scope_redirected_from", "duplicate_of",
                  "contradicts", "entities", "degraded", "facts_extracted"):
        assert field in declared, f"{field} would be silently stripped from the response"


def test_remember_response_declares_everything_writeresult_exposes():
    # A field added to WriteResult and forgotten here is invisible to every API caller.
    from synapse.core.write_pipeline import WriteResult
    from synapse.models.api import RememberResponse

    internal_only = {"confidence", "source", "reference_time"}  # not part of the public surface
    missing = set(WriteResult.model_fields) - set(RememberResponse.model_fields) - internal_only
    assert not missing, f"RememberResponse is missing WriteResult fields: {sorted(missing)}"
