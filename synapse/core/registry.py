"""Canonical registry of projects connected to Synapse (shared by API + scripts).

Projects are loaded from a JSON file (path from ``settings.projects_file``, defaulting to
``projects.json`` at the repo root). When that file is absent the registry falls back to
``projects.example.json`` with a logged warning so the app still starts cleanly.

The connector and the wiring scripts both import from here. The public API (``PROJECTS``,
``all_projects``, ``folder_name``, ``project_folder``, ``pretty_scope``) is unchanged.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from synapse.config import settings

logger = logging.getLogger("synapse.registry")

# ---------------------------------------------------------------------------
# Load the static project list from the configured JSON file
# ---------------------------------------------------------------------------

def _load_projects_file() -> dict[str, dict]:
    primary = Path(settings.projects_file)
    fallback = primary.parent / "projects.example.json"

    for path, is_fallback in [(primary, False), (fallback, True)]:
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if is_fallback:
                    logger.warning(
                        "projects.json not found at %s — using example file %s. "
                        "Copy projects.example.json to projects.json and fill in your project paths.",
                        primary, path,
                    )
                return data
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("could not read projects file %s: %s", path, exc)

    logger.warning(
        "No projects file found (looked for %s and %s). Registry will be empty.",
        primary, fallback,
    )
    return {}


PROJECTS: dict[str, dict] = _load_projects_file()


# ── Persistent overlay: projects added via the UI (not in the static list above) ──────────
# Lives under the projects root (dual host/container path: /projects in the container,
# <projects_root> on the host) so it persists across restarts and both see it.

def _overlay_path() -> Path | None:
    if not settings.projects_root:
        return None
    return Path(settings.projects_root) / ".synapse-connected.json"


def load_overlay() -> dict:
    p = _overlay_path()
    if p is None or not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("could not read connected-projects overlay: %s", exc)
    return {}


def add_connected(project_id: str, name: str, host_path: str, cluster: str = "added") -> None:
    """Persist a UI-added project so it survives restarts and keeps its real name/path."""
    p = _overlay_path()
    if p is None:
        logger.warning(
            "projects_root is not configured — cannot persist connected project %s", project_id
        )
        return
    overlay = load_overlay()
    overlay[project_id] = {"name": name, "path": host_path, "cluster": cluster,
                           "desc": f"{name} — connected via the UI."}
    p.write_text(json.dumps(overlay, indent=2), encoding="utf-8")


def all_projects() -> dict:
    """Static registry first, then UI-added overlay. Curated entries always win on id collisions."""
    overlay = {k: v for k, v in load_overlay().items() if k not in PROJECTS}
    return {**PROJECTS, **overlay}


def folder_name(host_path: str) -> str:
    """Last path segment, robust to Windows backslashes even when parsed on Linux."""
    return re.split(r"[\\/]", host_path.rstrip("\\/"))[-1]


def project_folder(host_path: str) -> Path:
    """The IO path for a project under the (host or container) projects_root.

    On the host: <projects_root>/<folder>. In the API container: /projects/<folder>
    (the mounted host projects dir). Derived from the folder name, so it's correct on both.
    """
    root = settings.projects_root or "."
    return Path(root) / folder_name(host_path)


def pretty_scope(scope: str) -> str:
    if scope == "global":
        return "global"
    return scope.replace("project_", "").replace("agent_", "agent-")
