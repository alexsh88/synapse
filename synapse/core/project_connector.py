"""ProjectConnector — wire a project to Synapse (Phase 11/12, shared by API + scripts).

All file ops are MERGE-safe + idempotent (never clobber an existing .mcp.json / settings.json /
CLAUDE.md). Command strings written into those files use the HOST synapse path
(settings.synapse_host_dir) so Claude Code on the host runs the right venv/scripts — even when this
connector runs inside the API container. Deep-seed reads a project's docs and feeds them through the
write pipeline; it reports progress via an injected callback (keeps this core module API-agnostic).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Awaitable, Callable

from synapse.config import settings

_HOOK_MARK = "session_brief.py"
_BLOCK_MARK = "<!-- synapse:integration -->"
ProgressCb = Callable[[int, int, int, str | None], Awaitable[None]] | None

# Docs read for deep-seed, in priority order.
_DOC_CANDIDATES = ["CLAUDE.md", "README.md", "ARCHITECTURE.md"]
_MAX_CHUNKS = 40


class ProjectConnector:
    def __init__(self) -> None:
        host = settings.synapse_host_dir.rstrip("/")
        self.venv_py = f"{host}/.venv/Scripts/python.exe"
        self.hook_script = f"{host}/scripts/session_brief.py"
        self.pythonpath = host

    # --- .mcp.json -----------------------------------------------------------

    def _mcp_server(self, project_id: str) -> dict:
        return {
            "command": self.venv_py,
            "args": ["-m", "synapse.mcp.server"],
            "env": {"SYNAPSE_PROJECT_ID": project_id, "PYTHONPATH": self.pythonpath},
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
## Synapse — shared knowledge brain

This project is connected to **Synapse**, the cross-project temporal knowledge graph
(`SYNAPSE_PROJECT_ID={project_id}`). Use the `synapse` MCP tools:

- **Session start:** `synapse:brief` loads this project's decisions, conventions, lessons, and
  relevant cross-project knowledge.
- **Learned something durable** (decision+rationale, convention, lesson, research, pattern, tool):
  `synapse:remember`. Store knowledge, never transcripts/scratch.
- **Need prior context:** `synapse:recall` (this project + global) or `synapse:search` (everything).

Scope is automatic: writes land in `project_{project_id}`; cross-cutting knowledge goes to `global`.
<!-- /synapse:integration -->
"""

    def write_claude_block(self, folder: Path, project_id: str, name: str) -> str:
        path = folder / "CLAUDE.md"
        block = self._claude_block(project_id, name)
        if path.exists():
            text = path.read_text(encoding="utf-8")
            if _BLOCK_MARK in text:
                return "CLAUDE.md block already present"
            path.write_text(text.rstrip() + "\n\n---\n\n" + block, encoding="utf-8")
            return "appended Synapse block to CLAUDE.md"
        path.write_text(f"# {name}\n\n" + block, encoding="utf-8")
        return "created CLAUDE.md"

    # --- SessionStart brief hook --------------------------------------------

    def install_hook(self, folder: Path, project_id: str) -> str:
        claude = folder / ".claude"
        claude.mkdir(exist_ok=True)
        path = claude / "settings.json"
        data: dict = {}
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return f"ABORT: {path} is invalid JSON"
        sessions = data.setdefault("hooks", {}).setdefault("SessionStart", [])
        for grp in sessions:
            for h in grp.get("hooks", []):
                if _HOOK_MARK in str(h.get("command", "")):
                    return "brief hook already installed"
        command = f"{self.venv_py} {self.hook_script} {project_id}"
        sessions.append({"hooks": [{"type": "command", "command": command}]})
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        return "installed brief hook"

    def write_files(self, folder: Path, project_id: str, name: str) -> list[str]:
        return [
            self.write_mcp(folder, project_id),
            self.write_claude_block(folder, project_id, name),
            self.install_hook(folder, project_id),
        ]

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

    def _chunks(self, folder: Path) -> list[str]:
        out: list[str] = []
        for doc in _DOC_CANDIDATES:
            path = folder / doc
            if not path.exists():
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
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
