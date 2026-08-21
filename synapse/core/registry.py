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


# Clusters that exist only as registry bookkeeping, not as a shared knowledge domain. A project
# in one of these gets no cluster tier at retrieval time (global + project only).
_NON_DOMAIN_CLUSTERS = frozenset({"", "added", "standalone", "none"})


def cluster_of(project_id: str) -> str | None:
    """The domain cluster a project belongs to, or ``None`` if it has no shared domain.

    The map lives in the REGISTRY (``projects.json``, gitignored) rather than in code, so real
    project names never reach the public repo — the same reason the registry exists at all.
    Retrieval uses this to compose the ``cluster_*`` tier (research §0); see
    :meth:`synapse.core.schema.Scope.cluster`.
    """
    meta = all_projects().get(project_id)
    if not meta:
        return None
    cluster = str(meta.get("cluster") or "").strip().lower()
    return None if cluster in _NON_DOMAIN_CLUSTERS else cluster


def folder_name(host_path: str) -> str:
    """Last path segment, robust to Windows backslashes even when parsed on Linux."""
    return re.split(r"[\\/]", host_path.rstrip("\\/"))[-1]


def project_roots() -> list[Path]:
    """Every directory a project folder may live under, primary first.

    The primary root is special beyond ordering: it owns the connected-projects overlay, and it is
    where an unresolvable project is reported against. Extras exist because projects don't all
    share one parent directory — and since the container reaches a host directory only through a
    bind mount, an out-of-root project arrives as its own root rather than a subdirectory.
    """
    roots = [settings.projects_root or "."]
    # Blank segments (a stray comma in the env var) would otherwise become Path("."), silently
    # resolving every project against the process working directory.
    roots += [r.strip() for r in settings.extra_project_roots.split(",") if r.strip()]
    return [Path(r) for r in roots]


def project_folder(host_path: str) -> Path:
    """The IO path for a project, resolved against each configured root in order.

    On the host: <root>/<folder>. In the API container: /projects/<folder> or an extra mount.
    Derived from the folder name, so it's correct on both.

    Falls back to <primary>/<folder> when the folder exists under no root, because the project
    LIST asks for a path in order to report ``exists: false`` about it — callers that need the
    absence to be an error check it themselves.
    """
    name = folder_name(host_path)
    roots = project_roots()
    for root in roots:
        if (candidate := root / name).is_dir():
            return candidate
    return roots[0] / name


def pretty_scope(scope: str) -> str:
    if scope == "global":
        return "global"
    return scope.replace("project_", "").replace("agent_", "agent-")
