"""ProjectConnector — wire a project to Synapse (Phase 11/12, shared by API + scripts).

All file ops are MERGE-safe + idempotent (never clobber an existing .mcp.json / settings.json /
CLAUDE.md). Command strings written into those files use the HOST synapse path
(settings.synapse_host_dir) so Claude Code on the host runs the right venv/scripts — even when this
connector runs inside the API container. Deep-seed reads a project's docs and feeds them through the
write pipeline; it reports progress via an injected callback (keeps this core module API-agnostic).
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Awaitable, Callable

from synapse.config import settings

_HOOK_MARK = "session_brief.py"
_RECALL_HOOK_MARK = "prompt_recall.py"
_BLOCK_MARK = "<!-- synapse:integration -->"
_BLOCK_END = "<!-- /synapse:integration -->"
# Bump when _claude_block changes materially; write_claude_block upgrades older blocks in place.
_BLOCK_VERSION = "<!-- synapse:block-v2 -->"
ProgressCb = Callable[[int, int, int, str | None], Awaitable[None]] | None

# Docs read for deep-seed. These three lead — they are where a project states itself — and any
# OTHER top-level markdown follows, because plenty of projects keep their substance elsewhere
# (BUILD-PLAN.md, PROMPTS.md, DESIGN.md). _MAX_CHUNKS truncates from the END, so this order is
# what survives a doc-heavy repo.
_DOC_PRIORITY = ["CLAUDE.md", "README.md", "ARCHITECTURE.md"]
# Standardised OSS boilerplate: the same text in every repo, so it says nothing about THIS project
# and would just spend the chunk budget (R2).
_DOC_SKIP = {"LICENSE.md", "CHANGELOG.md", "CONTRIBUTING.md", "CODE_OF_CONDUCT.md", "SECURITY.md"}
_MAX_CHUNKS = 40

# "C:/..." or "C:\..." — a drive letter is the one unambiguous tell that a path is Windows.
_WIN_DRIVE_RE = re.compile(r"^[A-Za-z]:[/\\]")


def _venv_python(host: str, *, host_dir_configured: bool) -> str:
    """Absolute path to the venv interpreter ON THE HOST.

    Windows venvs put the interpreter at ``.venv/Scripts/python.exe``, POSIX at ``.venv/bin/python``.
    Guessing wrong breaks every hook and the MCP server at once, and Claude Code reports it as
    nothing but ``-32000`` — so this is worth deriving rather than hardcoding.

    This process's own platform cannot answer the question. The connector normally runs inside the
    Linux API container, where ``os.name`` is always ``posix``, while the commands it writes are run
    by Claude Code on the host — which may be Windows. The configured host path is the signal that
    survives that boundary: a leading drive letter means Windows. Only when no host dir was
    configured are we necessarily wiring from the host itself, and then this process IS the host.
    """
    windows = bool(_WIN_DRIVE_RE.match(host)) if host_dir_configured else os.name == "nt"
    return f"{host}/.venv/Scripts/python.exe" if windows else f"{host}/.venv/bin/python"


class ProjectConnector:
    def __init__(self, *, in_container: bool | None = None) -> None:
        # Host path of this repo, written verbatim into project hook/.mcp commands so Claude Code
        # (on the host) runs the right venv/scripts. The env override is REQUIRED inside the API
        # container (code lives at /app there); when wiring from the host it's optional — fall back
        # to this repo's real location, which already IS the host path. Never leave it empty, or the
        # command becomes "/.venv/Scripts/python.exe" and the hook fails with "No such file".
        host = (settings.synapse_host_dir.rstrip("/")
                or str(Path(__file__).resolve().parents[2]).replace("\\", "/"))
        # "REQUIRED" above was only ever a comment. Nothing set the variable in the container, so
        # the fallback silently wrote /app into two projects' wiring and Claude Code reported it as
        # nothing but `-32000`. This is the one case where the fallback CANNOT be right, so name it.
        self._host_dir_unset = not settings.synapse_host_dir
        # Derived, not hardcoded: the same repo has to wire correctly from a Windows host and a
        # macOS/Linux one, and the container in between is Linux either way.
        self.venv_py = _venv_python(host, host_dir_configured=not self._host_dir_unset)
        self.hook_script = f"{host}/scripts/session_brief.py"
        self.recall_script = f"{host}/scripts/prompt_recall.py"
        self.pythonpath = host
        self._in_container = Path("/.dockerenv").exists() if in_container is None else in_container

    # --- .mcp.json -----------------------------------------------------------

    def _mcp_server(self, project_id: str) -> dict:
        return {
            "command": self.venv_py,
            "args": ["-m", "synapse.mcp.server"],
            "env": {
                "SYNAPSE_PROJECT_ID": project_id,
                "PYTHONPATH": self.pythonpath,
                # `python -m` prepends the CWD to sys.path, and the CWD here is the connected
                # project — so a project owning a top-level directory named like one of ours
                # SHADOWS it and the server cannot start at all. acme-sim does exactly that: it has
                # its own `synapse/` package (its API client), which won the import and produced
                # `ModuleNotFoundError: synapse.mcp.server` — a server that never came up, reported
                # to the user only as `Failed to reconnect to synapse: -32000`.
                #
                # PYTHONSAFEPATH (3.11+) suppresses just that prepend; PYTHONPATH still applies, so
                # the real package resolves. Set for EVERY project, not only acme-sim: the collision
                # depends on what a project happens to add later, so it must not be opt-in.
                "PYTHONSAFEPATH": "1",
            },
        }

    def write_mcp(self, folder: Path, project_id: str) -> str:
        path = folder / ".mcp.json"
        existed = path.exists()
        config: dict = {}
        if existed:
            try:
                config = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return f"ABORT: {path} is invalid JSON"
        servers = config.setdefault("mcpServers", {})
        want = self._mcp_server(project_id)
        if servers.get("synapse") == want:
            return ".mcp.json already correct"
        other = [k for k in servers if k != "synapse"]
        servers["synapse"] = want
        path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        return f"{'merged into' if existed else 'wrote'} .mcp.json" + (
            f" (kept: {', '.join(other)})" if other else "")

    # --- CLAUDE.md block -----------------------------------------------------

    def _claude_block(self, project_id: str, name: str) -> str:
        return f"""{_BLOCK_MARK}
{_BLOCK_VERSION}
## Synapse — shared knowledge brain (use PROACTIVELY)

This project is connected to **Synapse**, the cross-project temporal knowledge graph
(`SYNAPSE_PROJECT_ID={project_id}`). The session-start brief arrives automatically via hook;
everything else is on you — follow these trigger rules:

**RECALL before acting** (`synapse:recall`, or `synapse:search` for cross-project):
- Before designing a feature, refactoring, or making an architectural choice — a past
  decision may already settle it.
- When debugging anything non-obvious — a lesson for this component or error may exist.
- Before adding a dependency or changing a convention — check for prior art and prior rejections.

**REMEMBER immediately, not at session end** (`synapse:remember`):
- A decision is settled with rationale -> store it the moment it's made.
- A bug's root cause taught something durable -> store the lesson.
- A convention is established or changed -> store it.
- Quality bar: decisions, conventions, lessons, research, patterns, tools. NEVER transcripts,
  scratch work, or intermediate reasoning.

**CORRECT, don't ignore:** when recalled knowledge turns out wrong or outdated, call
`synapse:update` (supersedes with history) instead of silently working around it.

Scope is automatic: writes land in `project_{project_id}`; cross-cutting knowledge goes to `global`.
{_BLOCK_END}
"""

    def write_claude_block(self, folder: Path, project_id: str, name: str) -> str:
        path = folder / "CLAUDE.md"
        block = self._claude_block(project_id, name)
        if path.exists():
            text = path.read_text(encoding="utf-8")
            if _BLOCK_MARK in text:
                if _BLOCK_VERSION in text:
                    return "CLAUDE.md block already current"
                upgraded = re.sub(
                    re.escape(_BLOCK_MARK) + r".*?" + re.escape(_BLOCK_END),
                    block.strip(), text, count=1, flags=re.S,
                )
                path.write_text(upgraded, encoding="utf-8")
                return "upgraded Synapse block in CLAUDE.md"
            path.write_text(text.rstrip() + "\n\n---\n\n" + block, encoding="utf-8")
            return "appended Synapse block to CLAUDE.md"
        path.write_text(f"# {name}\n\n" + block, encoding="utf-8")
        return "created CLAUDE.md"

    # --- hooks (SessionStart brief + UserPromptSubmit recall) ----------------

    def _install_hook(self, folder: Path, event: str, script: str,
                      mark: str, project_id: str, label: str) -> str:
        """Merge a `<venv> <script> <project_id>` command into .claude/settings.json under
        `event`, preserving any other hooks. Idempotent (keyed on the script filename `mark`)."""
        claude = folder / ".claude"
        claude.mkdir(exist_ok=True)
        path = claude / "settings.json"
        data: dict = {}
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return f"ABORT: {path} is invalid JSON"
        groups = data.setdefault("hooks", {}).setdefault(event, [])
        command = f"{self.venv_py} {script} {project_id}"
        for grp in groups:
            for h in grp.get("hooks", []):
                if mark in str(h.get("command", "")):
                    if h.get("command") == command:
                        return f"{label} hook already installed"
                    # Matching on the script FILENAME alone reported "already installed" for a
                    # command naming a python.exe that doesn't exist on this machine, so no repair
                    # pass could ever reach it. write_mcp compares the whole entry; so does this.
                    h["command"] = command
                    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
                    return f"updated {label} hook"
        groups.append({"hooks": [{"type": "command", "command": command}]})
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        return f"installed {label} hook"

    def install_hook(self, folder: Path, project_id: str) -> str:
        return self._install_hook(folder, "SessionStart", self.hook_script,
                                  _HOOK_MARK, project_id, "brief")

    def install_recall_hook(self, folder: Path, project_id: str) -> str:
        return self._install_hook(folder, "UserPromptSubmit", self.recall_script,
                                  _RECALL_HOOK_MARK, project_id, "recall")

    def write_files(self, folder: Path, project_id: str, name: str) -> list[str]:
        actions = [
            self.write_mcp(folder, project_id),
            self.write_claude_block(folder, project_id, name),
            self.install_hook(folder, project_id),
            self.install_recall_hook(folder, project_id),
        ]
        if self._host_dir_unset and self._in_container:
            actions.insert(0, f"WARN: SYNAPSE_HOST_DIR is unset, so this wiring points inside the "
                              f"container ({self.pythonpath}) — Claude Code runs on the host and "
                              f"will fail to start the MCP server")
        return actions

    # --- status --------------------------------------------------------------

    def status(self, folder: Path) -> dict:
        mcp = folder / ".mcp.json"
        connected = hook = False
        if mcp.exists():
            try:
                connected = "synapse" in json.loads(mcp.read_text(encoding="utf-8")).get("mcpServers", {})
            except json.JSONDecodeError:
                pass
        settings_path = folder / ".claude" / "settings.json"
        if settings_path.exists():
            try:
                data = json.loads(settings_path.read_text(encoding="utf-8"))
                hook = any(_HOOK_MARK in str(h.get("command", ""))
                           for grp in data.get("hooks", {}).get("SessionStart", [])
                           for h in grp.get("hooks", []))
            except json.JSONDecodeError:
                pass
        return {"exists": folder.exists(), "connected": connected, "hook": hook}

    # --- seeding -------------------------------------------------------------

    async def seed_entity(self, engine, project_id: str, description: str) -> str:
        r = await engine.remember(description, project_id=project_id, source="connect", force=True)
        return r.outcome.value

    def _docs(self, folder: Path) -> list[Path]:
        """Top-level markdown, priority names first, then the rest sorted for determinism.

        NOT recursive, deliberately: a repo's node_modules/ or docs/ tree would swamp the chunk
        budget with vendored or reference text that isn't this project stating itself.
        """
        named = [folder / name for name in _DOC_PRIORITY]
        try:
            rest = sorted(p for p in folder.glob("*.md")
                          if p.name not in _DOC_PRIORITY and p.name not in _DOC_SKIP)
        except OSError:                                   # unreadable folder — seed what we named
            rest = []
        return [p for p in named + rest if p.is_file()]

    def _chunks(self, folder: Path) -> list[str]:
        out: list[str] = []
        for path in self._docs(folder):
            text = path.read_text(encoding="utf-8", errors="ignore")
            # Drop OUR OWN block first. write_files puts it into CLAUDE.md, so without this every
            # connect seeds Synapse's boilerplate into the project's scope — identical text in
            # every connected project, which is exactly the noise R2 exists to keep out.
            text = re.sub(re.escape(_BLOCK_MARK) + r".*?" + re.escape(_BLOCK_END), "",
                          text, flags=re.DOTALL)
            text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)  # drop code fences
            for para in re.split(r"\n\s*\n", text):
                p = re.sub(r"\s+", " ", para).strip().lstrip("#").strip()
                if len(p) >= 80 and not p.startswith(("|", "-", "*", ">")):
                    out.append(p)
        return out[:_MAX_CHUNKS]

    async def deep_seed(self, engine, project_id: str, folder: Path,
                        on_progress: ProgressCb = None) -> dict:
        """Feed the project's docs through the write pipeline. Triage + dedup guard quality."""
        chunks = self._chunks(folder)
        total, stored = len(chunks), 0
        for i, chunk in enumerate(chunks, 1):
            try:
                r = await engine.remember(chunk, project_id=project_id, source="connect")
                ok = r.outcome.value in ("stored", "contradiction")
                stored += ok
                if on_progress:
                    await on_progress(i, total, stored, r.facts[0] if (ok and r.facts) else None)
            except Exception:  # noqa: BLE001 — one bad chunk shouldn't abort the whole seed
                if on_progress:
                    await on_progress(i, total, stored, None)
        return {"chunks": total, "stored": stored}
