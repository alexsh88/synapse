"""ProjectConnector — merge-safe file ops, status, deep-seed. No live services."""

from __future__ import annotations

import json
from types import SimpleNamespace

from synapse.config import settings
from synapse.core.project_connector import ProjectConnector


def _c() -> ProjectConnector:
    return ProjectConnector()


def test_write_mcp_merges_and_is_idempotent(tmp_path):
    c = _c()
    # pre-existing .mcp.json with another server must be preserved.
    (tmp_path / ".mcp.json").write_text(json.dumps({"mcpServers": {"postgres": {"command": "x"}}}), encoding="utf-8")
    a1 = c.write_mcp(tmp_path, "acme-store")
    data = json.loads((tmp_path / ".mcp.json").read_text(encoding="utf-8"))
    assert "postgres" in data["mcpServers"] and "synapse" in data["mcpServers"]
    assert data["mcpServers"]["synapse"]["env"]["SYNAPSE_PROJECT_ID"] == "acme-store"
    # host command path comes from settings.synapse_host_dir (not the container /app)
    assert settings.synapse_host_dir in data["mcpServers"]["synapse"]["command"]
    assert "merged" in a1
    assert "already correct" in c.write_mcp(tmp_path, "acme-store")  # idempotent


def test_install_hook_merges_preserving_other_settings(tmp_path):
    c = _c()
    (tmp_path / ".claude").mkdir()
    (tmp_path / ".claude" / "settings.json").write_text(
        json.dumps({"effortLevel": "high", "hooks": {"PreToolUse": [{"matcher": "Bash", "hooks": []}]}}),
        encoding="utf-8")
    assert "installed" in c.install_hook(tmp_path, "acme-flow")
    data = json.loads((tmp_path / ".claude" / "settings.json").read_text(encoding="utf-8"))
    assert data["effortLevel"] == "high"                       # preserved
    assert "PreToolUse" in data["hooks"]                       # preserved
    cmd = data["hooks"]["SessionStart"][0]["hooks"][0]["command"]
    assert "session_brief.py acme-flow" in cmd
    assert "already installed" in c.install_hook(tmp_path, "acme-flow")  # idempotent


def test_claude_block_created_then_appended(tmp_path):
    c = _c()
    assert "created" in c.write_claude_block(tmp_path, "acme-bot", "Acme-Bot")
    assert "Synapse" in (tmp_path / "CLAUDE.md").read_text(encoding="utf-8")
    assert "already present" in c.write_claude_block(tmp_path, "acme-bot", "Acme-Bot")
    # an existing CLAUDE.md gets the block appended, original kept
    p = tmp_path / "sub"; p.mkdir()
    (p / "CLAUDE.md").write_text("# My Project\n\nexisting content\n", encoding="utf-8")
    c.write_claude_block(p, "x", "X")
    text = (p / "CLAUDE.md").read_text(encoding="utf-8")
    assert "existing content" in text and "synapse:integration" in text


def test_status_reflects_files(tmp_path):
    c = _c()
    assert c.status(tmp_path) == {"exists": True, "connected": False, "hook": False}
    c.write_mcp(tmp_path, "x")
    c.install_hook(tmp_path, "x")
    assert c.status(tmp_path) == {"exists": True, "connected": True, "hook": True}


class _FakeResult:
    def __init__(self, outcome, facts=None):
        self.outcome = SimpleNamespace(value=outcome)
        self.facts = facts or []


class _FakeEngine:
    def __init__(self):
        self.calls = []

    async def remember(self, content, *, project_id=None, source=None, force=False, **kw):
        self.calls.append((content, project_id, force))
        # pretend every other chunk is novel
        return _FakeResult("stored" if len(self.calls) % 2 else "duplicate", facts=["a fact"])


async def test_deep_seed_chunks_docs_and_reports_progress(tmp_path):
    (tmp_path / "CLAUDE.md").write_text(
        "# Proj\n\n" + "\n\n".join(f"This is durable paragraph number {i} with enough length to pass the "
                                   f"eighty character minimum filter for chunking." for i in range(5)),
        encoding="utf-8")
    engine = _FakeEngine()
    events = []

    async def on_progress(done, total, stored, fact):
        events.append((done, total, stored))

    result = await _c().deep_seed(engine, "proj", tmp_path, on_progress)
    assert result["chunks"] == 5 and result["stored"] >= 1
    assert len(events) == 5 and events[-1][0] == 5          # progress fired per chunk
    assert all(c[1] == "proj" for c in engine.calls)        # scoped to the project
