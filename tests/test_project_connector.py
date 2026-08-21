"""ProjectConnector — merge-safe file ops, status, deep-seed. No live services."""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

from synapse.config import settings
from synapse.core.project_connector import ProjectConnector

# With no configured host dir the connector is wiring from the host itself, so the interpreter
# layout is this machine's. Tests that exercise that fallback must therefore expect THIS platform
# or they pass on Windows and fail on Linux CI. Tests that pin the cross-platform behaviour pass an
# explicit host path instead (see the two below), which is machine-independent by construction.
_FALLBACK_VENV = "/.venv/Scripts/python.exe" if os.name == "nt" else "/.venv/bin/python"


def _c() -> ProjectConnector:
    return ProjectConnector()


# --- the host-path override is REQUIRED in a container, so say so when it's missing ---------
#
# The connector falls back to its OWN on-disk location when synapse_host_dir is unset. On the host
# that IS the host path; inside the API container it is /app, so the wiring it writes names a
# python.exe that does not exist on the host and Claude Code reports only `-32000`. Two projects
# (acme-prep, acme-mobile) shipped that way — both connected through the UI, both
# silent about it — before this check existed.


def test_wiring_written_from_a_container_without_the_override_says_it_is_broken(tmp_path,
                                                                                monkeypatch):
    monkeypatch.setattr(settings, "synapse_host_dir", "")
    actions = ProjectConnector(in_container=True).write_files(tmp_path, "demo", "Demo")
    assert any(a.startswith("WARN") and "SYNAPSE_HOST_DIR" in a for a in actions), actions


def test_a_configured_host_path_is_not_flagged(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "synapse_host_dir", "C:/repo/synapse")
    actions = ProjectConnector(in_container=True).write_files(tmp_path, "demo", "Demo")
    assert not any(a.startswith("WARN") for a in actions), actions


def test_wiring_from_the_host_needs_no_override(tmp_path, monkeypatch):
    """Unset is legitimate on the host — the fallback is already the right path there, so this
    must not become a warning everyone learns to ignore."""
    monkeypatch.setattr(settings, "synapse_host_dir", "")
    actions = ProjectConnector(in_container=False).write_files(tmp_path, "demo", "Demo")
    assert not any(a.startswith("WARN") for a in actions), actions


def test_a_stale_hook_command_is_corrected_not_reported_as_already_installed(tmp_path,
                                                                             monkeypatch):
    """Idempotency was keyed on the script FILENAME, so a hook naming a python.exe that does not
    exist on this machine stayed "already installed" forever and no repair pass could reach it.
    .mcp.json compares the whole entry and self-heals; the hooks have to as well — otherwise
    re-connecting to fix broken wiring silently does half the job."""
    monkeypatch.setattr(settings, "synapse_host_dir", "/app")            # as written in-container
    ProjectConnector(in_container=False).install_hook(tmp_path, "demo")

    monkeypatch.setattr(settings, "synapse_host_dir", "C:/repo/synapse")  # the real host path
    action = ProjectConnector(in_container=False).install_hook(tmp_path, "demo")

    data = json.loads((tmp_path / ".claude" / "settings.json").read_text(encoding="utf-8"))
    commands = [h["command"] for g in data["hooks"]["SessionStart"] for h in g["hooks"]]
    assert commands == ["C:/repo/synapse/.venv/Scripts/python.exe "
                        "C:/repo/synapse/scripts/session_brief.py demo"], commands
    assert "updated" in action, action


def test_an_identical_hook_is_still_left_alone(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "synapse_host_dir", "C:/repo/synapse")
    c = ProjectConnector(in_container=False)
    c.install_hook(tmp_path, "demo")
    assert "already installed" in c.install_hook(tmp_path, "demo")


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


def test_write_mcp_protects_against_a_project_shadowing_our_package(tmp_path):
    """`python -m` prepends the CWD, so a project directory named `synapse/` wins the import.

    acme-sim has its own `synapse/` package (its API client). Without PYTHONSAFEPATH the MCP server
    could not start there at all — `ModuleNotFoundError: synapse.mcp.server`, surfaced to the user
    only as `Failed to reconnect to synapse: -32000`. Every project gets the guard, because whether
    a collision exists depends on what that project adds later.
    """
    c = _c()
    c.write_mcp(tmp_path, "acme-sim")
    env = json.loads((tmp_path / ".mcp.json").read_text(encoding="utf-8"))["mcpServers"]["synapse"]["env"]
    assert env["PYTHONSAFEPATH"] == "1"
    # PYTHONSAFEPATH only suppresses the CWD prepend — the real package must still be findable.
    assert env["PYTHONPATH"]


def test_the_mcp_command_is_never_rooted_at_the_filesystem_root(tmp_path):
    """Regression pin for edc9b33: an empty host dir produced `/.venv/Scripts/python.exe`.

    That commit fixed the connector but not the 9 already-written files, which stayed broken until
    2026-07-27 — so the pin is on the produced VALUE, not on the fallback logic.
    """
    c = _c()
    c.write_mcp(tmp_path, "acme-flow")
    server = json.loads((tmp_path / ".mcp.json").read_text(encoding="utf-8"))["mcpServers"]["synapse"]
    assert not server["command"].startswith("/."), server["command"]
    assert server["command"].endswith(_FALLBACK_VENV)
    assert len(server["command"]) > len(_FALLBACK_VENV)


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


def test_empty_host_dir_falls_back_to_repo_root_never_leading_slash(monkeypatch):
    # Regression: an empty synapse_host_dir must NOT yield "/.venv/Scripts/python.exe"
    # (which fails at runtime with "No such file"); it falls back to this repo's location.
    monkeypatch.setattr(settings, "synapse_host_dir", "")
    c = _c()
    assert not c.venv_py.startswith("/.venv")
    assert c.venv_py.endswith(_FALLBACK_VENV)
    assert c.recall_script.endswith("/scripts/prompt_recall.py")
    assert "synapse" in c.venv_py.lower()


# --- the interpreter layout follows the HOST, which is not this process ----------------------
#
# The connector runs in the Linux API container but writes commands Claude Code executes on the
# host. Hardcoding `.venv/Scripts/python.exe` therefore worked only as long as every host was
# Windows; on macOS it names a file that cannot exist, and Claude Code reports that as `-32000`
# with nothing else to go on. The configured host path is the one signal that crosses the
# container boundary, so both cases are pinned explicitly rather than by platform.


def test_a_windows_host_path_gets_the_windows_interpreter(monkeypatch):
    monkeypatch.setattr(settings, "synapse_host_dir", "C:/Users/dev/synapse")
    assert _c().venv_py == "C:/Users/dev/synapse/.venv/Scripts/python.exe"


def test_a_posix_host_path_gets_the_posix_interpreter(monkeypatch):
    monkeypatch.setattr(settings, "synapse_host_dir", "/Users/dev/synapse")
    assert _c().venv_py == "/Users/dev/synapse/.venv/bin/python"


def test_recall_hook_merges_alongside_brief_hook(tmp_path):
    c = _c()
    # both hooks coexist in one settings.json under different events, preserving each other.
    assert "installed brief" in c.install_hook(tmp_path, "acme-flow")
    assert "installed recall" in c.install_recall_hook(tmp_path, "acme-flow")
    data = json.loads((tmp_path / ".claude" / "settings.json").read_text(encoding="utf-8"))
    brief_cmd = data["hooks"]["SessionStart"][0]["hooks"][0]["command"]
    recall_cmd = data["hooks"]["UserPromptSubmit"][0]["hooks"][0]["command"]
    assert "session_brief.py acme-flow" in brief_cmd
    assert "prompt_recall.py acme-flow" in recall_cmd
    assert "already installed" in c.install_recall_hook(tmp_path, "acme-flow")  # idempotent
    # write_files wires both hooks in one pass
    fresh = tmp_path / "fresh"; fresh.mkdir()
    actions = c.write_files(fresh, "x", "X")
    assert any("recall hook" in a for a in actions)
    data2 = json.loads((fresh / ".claude" / "settings.json").read_text(encoding="utf-8"))
    assert "UserPromptSubmit" in data2["hooks"] and "SessionStart" in data2["hooks"]


def test_claude_block_created_then_appended(tmp_path):
    c = _c()
    assert "created" in c.write_claude_block(tmp_path, "acme-bot", "Acme-Bot")
    assert "Synapse" in (tmp_path / "CLAUDE.md").read_text(encoding="utf-8")
    assert "already current" in c.write_claude_block(tmp_path, "acme-bot", "Acme-Bot")
    # an existing CLAUDE.md gets the block appended, original kept
    p = tmp_path / "sub"; p.mkdir()
    (p / "CLAUDE.md").write_text("# My Project\n\nexisting content\n", encoding="utf-8")
    c.write_claude_block(p, "x", "X")
    text = (p / "CLAUDE.md").read_text(encoding="utf-8")
    assert "existing content" in text and "synapse:integration" in text


def test_claude_block_upgrades_older_version_in_place(tmp_path):
    c = _c()
    old_block = (
        "<!-- synapse:integration -->\n## Synapse — shared knowledge brain\n"
        "old v1 wording\n<!-- /synapse:integration -->\n"
    )
    (tmp_path / "CLAUDE.md").write_text(
        "# My Project\n\nexisting content\n\n---\n\n" + old_block + "\ntrailing user notes\n",
        encoding="utf-8",
    )
    assert "upgraded" in c.write_claude_block(tmp_path, "x", "X")
    text = (tmp_path / "CLAUDE.md").read_text(encoding="utf-8")
    # v1 wording replaced by the current block; surrounding content untouched
    assert "old v1 wording" not in text
    assert "PROACTIVELY" in text and "synapse:block-v2" in text
    assert "existing content" in text and "trailing user notes" in text
    # second call is a no-op
    assert "already current" in c.write_claude_block(tmp_path, "x", "X")


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


# --- what deep-seed READS (2026-07-30) ----------------------------------------------------
#
# The doc list was three hardcoded names, so a project keeping its substance anywhere else got
# seeded with almost nothing: one connected this day held its content in BUILD-PLAN.md and
# PROMPTS.md, and the graph ended up with those two FILENAMES as entities and little else.


def _paras(tag: str, n: int) -> str:
    return "\n\n".join(f"{tag} paragraph {i}, written long enough to clear the eighty character "
                       f"floor that the chunker uses to skip noise." for i in range(n))


def test_deep_seed_reads_any_top_level_markdown(tmp_path):
    (tmp_path / "BUILD-PLAN.md").write_text(_paras("Plan", 2), encoding="utf-8")
    (tmp_path / "PROMPTS.md").write_text(_paras("Prompt", 2), encoding="utf-8")
    chunks = _c()._chunks(tmp_path)
    assert sum("Plan paragraph" in c for c in chunks) == 2
    assert sum("Prompt paragraph" in c for c in chunks) == 2


def test_deep_seed_does_not_re_ingest_synapses_own_claude_block(tmp_path):
    """`write_files` CREATES that block and `_chunks` then read it straight back, so every connect
    seeded Synapse's own boilerplate into the project's scope — the same text in all 11 (R2)."""
    (tmp_path / "CLAUDE.md").write_text("# Proj\n\n" + _paras("Real", 2), encoding="utf-8")
    _c().write_claude_block(tmp_path, "proj", "Proj")
    assert "Synapse" in (tmp_path / "CLAUDE.md").read_text(encoding="utf-8")   # block really there

    chunks = _c()._chunks(tmp_path)
    assert sum("Real paragraph" in c for c in chunks) == 2
    assert not [c for c in chunks if "Synapse" in c], "our own block leaked into the seed"


def test_deep_seed_ignores_nested_and_boilerplate_markdown(tmp_path):
    """Top level only: a repo's node_modules/ or docs/ tree would swamp the chunk budget with
    vendored text, and OSS boilerplate is identical everywhere so it says nothing about a project."""
    (tmp_path / "README.md").write_text(_paras("Readme", 1), encoding="utf-8")
    (tmp_path / "LICENSE.md").write_text(_paras("License", 2), encoding="utf-8")
    vendored = tmp_path / "node_modules" / "pkg"
    vendored.mkdir(parents=True)
    (vendored / "README.md").write_text(_paras("Vendored", 2), encoding="utf-8")

    chunks = _c()._chunks(tmp_path)
    assert sum("Readme paragraph" in c for c in chunks) == 1
    assert not [c for c in chunks if "License paragraph" in c or "Vendored paragraph" in c]


def test_the_priority_docs_are_chunked_first(tmp_path):
    """_MAX_CHUNKS truncates from the END, so ordering decides what survives on a doc-heavy repo."""
    (tmp_path / "ZZZ-appendix.md").write_text(_paras("Appendix", 1), encoding="utf-8")
    (tmp_path / "README.md").write_text(_paras("Readme", 1), encoding="utf-8")
    (tmp_path / "CLAUDE.md").write_text(_paras("Claude", 1), encoding="utf-8")
    chunks = _c()._chunks(tmp_path)
    assert [c.split()[0] for c in chunks] == ["Claude", "Readme", "Appendix"]
