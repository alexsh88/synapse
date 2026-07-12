"""Install Synapse's Claude Code hooks into projects' .claude/settings.json.

- **SessionStart** -> session_brief.py  (auto-loads the project's brief at session start; T11)
- **PreCompact** + **SessionEnd** -> capture_session.py  (auto-captures durable lessons; the capture
  feature). PreCompact catches long sessions before context loss; SessionEnd catches short ones.

MERGE-safe (preserves any existing settings + hooks/keys) and idempotent (skips a hook already
present, keyed by the script filename in its command).

    python -m scripts.install_brief_hook --all          # every registered project
    python -m scripts.install_brief_hook acme-api acme-flow   # specific
    python -m scripts.install_brief_hook --list          # show brief + capture status
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.wire_project import PROJECTS, SYNAPSE_DIR, VENV_PY

_BRIEF = (SYNAPSE_DIR / "scripts" / "session_brief.py").as_posix()
_CAPTURE = (SYNAPSE_DIR / "scripts" / "capture_session.py").as_posix()


def _brief_cmd(pid: str) -> str:
    return f"{VENV_PY} {_BRIEF} {pid}"


def _capture_cmd(pid: str) -> str:
    return f"{VENV_PY} {_CAPTURE} {pid}"


# (event, command builder, idempotency marker, extra entry fields)
_HOOKS = [
    ("SessionStart", _brief_cmd, "session_brief.py", {}),
    ("PreCompact", _capture_cmd, "capture_session.py", {"timeout": 30}),
    ("SessionEnd", _capture_cmd, "capture_session.py", {"timeout": 30}),
]


def _present(data: dict, event: str, marker: str) -> bool:
    for grp in data.get("hooks", {}).get(event, []):
        for h in grp.get("hooks", []):
            if marker in str(h.get("command", "")):
                return True
    return False


def _load(settings: Path) -> dict | None:
    if not settings.exists():
        return {}
    try:
        return json.loads(settings.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def install(project_id: str, path: Path) -> str:
    if not path.exists():
        return f"SKIP: path missing: {path}"
    claude = path / ".claude"
    claude.mkdir(exist_ok=True)
    settings = claude / "settings.json"
    data = _load(settings)
    if data is None:
        return f"ABORT: {settings} is invalid JSON — fix by hand, not overwriting"

    added = []
    for event, cmd_fn, marker, extra in _HOOKS:
        if _present(data, event, marker):
            continue
        entry = {"type": "command", "command": cmd_fn(project_id), **extra}
        data.setdefault("hooks", {}).setdefault(event, []).append({"hooks": [entry]})
        added.append(event)
    if not added:
        return "all hooks already installed"
    settings.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    other = [k for k in data if k != "hooks"]
    return f"installed {', '.join(added)}" + (f" (kept: {', '.join(other)})" if other else "")


def main() -> int:
    ap = argparse.ArgumentParser(description="Install Synapse's Claude Code hooks (brief + capture).")
    ap.add_argument("project_id", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list or (not args.project_id and not args.all):
        print(f"{'id':24} brief  capture")
        for pid, p in PROJECTS.items():
            data = _load(Path(p["path"]) / ".claude" / "settings.json") or {}
            brief = _present(data, "SessionStart", "session_brief.py")
            cap = _present(data, "PreCompact", "capture_session.py") or \
                _present(data, "SessionEnd", "capture_session.py")
            print(f"{pid:24} {'yes' if brief else 'no ':6} {'yes' if cap else 'no'}")
        return 0

    ids = list(PROJECTS) if args.all else args.project_id
    unknown = [i for i in ids if i not in PROJECTS]
    if unknown:
        print(f"[error] unknown: {', '.join(unknown)}")
        return 2
    for pid in ids:
        print(f"  {pid:24} {install(pid, Path(PROJECTS[pid]['path']))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
