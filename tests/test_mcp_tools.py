"""Unit tests for the MCP tool layer (plan Part 6).

Tools are tested against a fake engine: scope resolution, param mapping, result
shape, and error surfacing — no live services. A live exercise of all seven tools
lives in scripts/mcp_smoke.py.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from synapse.core.retrieval_engine import Brief, Recalled
from synapse.core.write_pipeline import Outcome, WriteResult
from synapse.mcp import tools as t

DEFAULT_PROJECT = "acme-store"


class FakeEngine:
    def __init__(self):
        self.calls: list[tuple] = []

    async def remember(self, content, *, knowledge_type=None, project_id=None, cluster=None,
                       force=False):
        self.calls.append(("remember", content, knowledge_type, project_id, cluster))
        return WriteResult(
            outcome=Outcome.STORED, knowledge_type=knowledge_type or "decision",
            scope=(f"cluster_{cluster}" if cluster
                   else "global" if project_id is None else f"project_{project_id}"),
            episode_uuid="ep-1", entities=["Acme-Store"], facts=["Acme-Store uses X"],
        )

    async def recall(self, query, *, project_id=None, limit=10, as_of=None,
                     feedback=False, **kw):
        self.calls.append(("recall", query, project_id, limit, as_of))
        return [Recalled(fact="a fact", score=0.9, scope="project_acme-store", uuid="u1")]

    async def brief(self, project_id):
        self.calls.append(("brief", project_id))
        return Brief(
            project_id=project_id, project_summary="summary", active_conventions=["c"],
            key_decisions=["d"], relevant_lessons=["l"], cross_project_knowledge=["x"],
            generated_at=datetime(2026, 5, 31, tzinfo=timezone.utc),
        )

    async def search(self, query, *, group_ids=None, limit=10, as_of=None):
        self.calls.append(("search", query, group_ids, limit, as_of))
        return [Recalled(fact="cross fact", score=0.8, scope="project_other", uuid="u2")]

    async def relate(self, from_id, to_id, rel):
        self.calls.append(("relate", from_id, to_id, rel))
        return {"success": True, "edge_uuid": "e1"}

    async def forget(self, knowledge_id, reason=None):
        self.calls.append(("forget", knowledge_id, reason))
        return {"success": True}

    async def update(self, knowledge_id, changes, *, project_id=None):
        self.calls.append(("update", knowledge_id, changes, project_id))
        return {"success": True, "new": {"outcome": "stored"}}


# --- remember ----------------------------------------------------------------


async def test_remember_defaults_to_project_scope():
    eng = FakeEngine()
    out = await t.remember(eng, DEFAULT_PROJECT, "we chose X")
    assert eng.calls[0] == ("remember", "we chose X", None, "acme-store", None)
    assert out["outcome"] == "stored" and out["scope"] == "project_acme-store"


async def test_remember_global_scope_and_type():
    eng = FakeEngine()
    out = await t.remember(eng, DEFAULT_PROJECT, "universal rule", type="convention", scope="global")
    assert eng.calls[0] == ("remember", "universal rule", "convention", None, None)
    assert out["scope"] == "global"


async def test_remember_folds_in_relationships():
    eng = FakeEngine()
    await t.remember(eng, DEFAULT_PROJECT, "fact", relationships="relates to Y")
    assert "relates to Y" in eng.calls[0][1]


# --- recall ------------------------------------------------------------------


async def test_recall_shapes_results_and_parses_as_of():
    eng = FakeEngine()
    out = await t.recall(eng, DEFAULT_PROJECT, "why X?", as_of="2026-03-01T00:00:00Z", limit=5)
    _, query, project_id, limit, as_of = eng.calls[0]
    assert project_id == "acme-store" and limit == 5
    assert as_of == datetime(2026, 3, 1, tzinfo=timezone.utc)
    assert out["count"] == 1 and out["results"][0]["id"] == "u1"


async def test_recall_global_scope():
    eng = FakeEngine()
    await t.recall(eng, DEFAULT_PROJECT, "q", scope="global")
    assert eng.calls[0][2] is None  # project_id -> None for global


# --- brief -------------------------------------------------------------------


async def test_brief_defaults_to_server_project():
    eng = FakeEngine()
    out = await t.brief(eng, DEFAULT_PROJECT)
    assert eng.calls[0] == ("brief", "acme-store")
    assert out["active_conventions"] == ["c"] and out["project_id"] == "acme-store"


async def test_brief_errors_without_project():
    eng = FakeEngine()
    out = await t.brief(eng, None)
    assert "error" in out and not eng.calls  # never reached the engine


# --- search ------------------------------------------------------------------


async def test_search_maps_scope_filter_and_reports_ignored():
    eng = FakeEngine()
    out = await t.search(eng, "monetization", filters={"scope": "project:mindtales", "type": "lesson"})
    assert eng.calls[0][2] == ["project_mindtales"]  # group_ids
    assert out["filters_ignored"] == ["type"]


async def test_search_all_scopes_by_default():
    eng = FakeEngine()
    await t.search(eng, "anything")
    assert eng.calls[0][2] is None  # group_ids None => all knowledge


# --- relate / forget / update ------------------------------------------------


async def test_relate_passthrough():
    eng = FakeEngine()
    out = await t.relate(eng, "a", "b", "shares_pattern")
    assert eng.calls[0] == ("relate", "a", "b", "shares_pattern") and out["success"]


async def test_forget_passthrough():
    eng = FakeEngine()
    out = await t.forget(eng, "fact-1", reason="obsolete")
    assert eng.calls[0] == ("forget", "fact-1", "obsolete") and out["success"]


async def test_update_passes_default_project():
    eng = FakeEngine()
    out = await t.update(eng, DEFAULT_PROJECT, "fact-1", {"content": "new truth"})
    assert eng.calls[0] == ("update", "fact-1", {"content": "new truth"}, "acme-store")
    assert out["success"]


# --- server registration -----------------------------------------------------


async def test_server_registers_the_full_tool_surface():
    from synapse.mcp.server import mcp

    names = {tool.name for tool in await mcp.list_tools()}
    # remember_runbook / runbooks are separate tools rather than a `type="runbook"` flag on
    # remember, because they take a different SHAPE of input: an ordered list instead of prose.
    # Folding them into remember would mean accepting prose and hoping extraction preserved the
    # ordering, which is the defect roadmap item 18 exists to fix.
    assert names == {
        "remember", "recall", "brief", "search", "relate", "forget", "update",
        "remember_runbook", "runbooks",
    }
