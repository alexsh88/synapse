"""Wire a project to Synapse — Phase 11 rollout (idempotent).

For one project it: (1) writes `.mcp.json` into the project root, (2) adds a Synapse
usage block to the project's `CLAUDE.md` (creating it if absent, guarded by a marker
so re-runs don't duplicate), (3) seeds an accurate Project-entity description into the
graph (force=True bulk import, bypassing the write-trigger filter), and (4) smoke-tests
`brief(project_id)` so you can see it load — including cross-project global knowledge (R5).

Project registry (id → name/path/cluster/description) is loaded from ``projects.json``
at the repo root (see ``synapse/core/registry.py``). To override, set ``PROJECTS_FILE``
in the environment or ``.env``.

    python -m scripts.wire_project acme-jobs            # one project
    python -m scripts.wire_project acme-jobs --no-seed  # just files, no graph write
    python -m scripts.wire_project --list                 # show registry + connection status
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

SYNAPSE_DIR = Path(__file__).resolve().parents[1]

_MARKER = "<!-- synapse:integration -->"


def _get_projects() -> dict[str, dict]:
    """Load the project registry from the JSON file (via synapse.core.registry)."""
    from synapse.core.registry import PROJECTS
    return PROJECTS


def _get_connector():
    from synapse.core.project_connector import ProjectConnector
    return ProjectConnector()


def _has_synapse(mcp_path: Path) -> bool:
    if not mcp_path.exists():
        return False
    try:
        return "synapse" in json.loads(mcp_path.read_text(encoding="utf-8")).get("mcpServers", {})
    except (json.JSONDecodeError, OSError):
        return False


def write_files(project_id: str, name: str, path: Path) -> list[str]:
    if not path.exists():
        return [f"SKIP: project path does not exist: {path}"]
    connector = _get_connector()
    return connector.write_files(path, project_id, name)


async def seed_and_verify(project_ids: list[str]) -> None:
    """Open ONE engine session and seed + smoke-test brief for each project."""
    from synapse.config import settings
    from synapse.core.knowledge_engine import KnowledgeEngine
    from synapse.mcp import tools as t

    if not settings.anthropic_api_key:
        print("  [warn] ANTHROPIC_API_KEY missing — skipping seed/verify")
        return

    projects = _get_projects()
    async with KnowledgeEngine() as engine:
        for pid in project_ids:
            desc = projects[pid]["desc"]
            r = await engine.remember(desc, project_id=pid, source="seed", force=True)
            b = await t.brief(engine, pid)
            print(f"  {pid}: seed [{r.outcome.value}] entities={r.entities[:2]} | brief -> "
                  f"dec={len(b['key_decisions'])} conv={len(b['active_conventions'])} "
                  f"less={len(b['relevant_lessons'])} cross={len(b['cross_project_knowledge'])}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Wire one or more projects to Synapse (Phase 11).")
    ap.add_argument("project_id", nargs="*", help="one or more ids, e.g. acme-jobs acme-flow")
    ap.add_argument("--all", action="store_true", help="wire every project not yet synapse-connected")
    ap.add_argument("--no-seed", action="store_true", help="write files only; no graph write / brief test")
    ap.add_argument("--list", action="store_true", help="list registry + connection status")
    args = ap.parse_args()

    projects = _get_projects()

    if args.list or (not args.project_id and not args.all):
        print(f"{'id':24} {'cluster':10} synapse?  path")
        for pid, p in projects.items():
            connected = _has_synapse(Path(p["path"]) / ".mcp.json")
            print(f"{pid:24} {p['cluster']:10} {'yes' if connected else 'no ':9} {p['path']}")
        return 0

    if args.all:
        ids = [pid for pid, p in projects.items() if not _has_synapse(Path(p["path"]) / ".mcp.json")]
        print(f"== --all: {len(ids)} not-yet-connected: {', '.join(ids) or '(none)'} ==")
    else:
        ids = args.project_id
    unknown = [pid for pid in ids if pid not in projects]
    if unknown:
        print(f"[error] unknown project(s): {', '.join(unknown)}. Known: {', '.join(projects)}")
        return 2

    seedable: list[str] = []
    for pid in ids:
        p = projects[pid]
        print(f"== wiring {pid} ({p['name']}) ==")
        actions = write_files(pid, p["name"], Path(p["path"]))
        for a in actions:
            print(f"  {a}")
        if not any(x.startswith(("SKIP", "ABORT")) for x in actions):
            seedable.append(pid)
    if seedable and not args.no_seed:
        print("== seeding + brief smoke (one engine session) ==")
        asyncio.run(seed_and_verify(seedable))
    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
