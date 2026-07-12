#!/usr/bin/env python
"""SessionStart hook helper — inject a project's Synapse brief as session context (T11).

Standalone (stdlib only, no synapse imports) so it runs under any Python without PYTHONPATH
fuss. Hits the running Synapse API's cached brief endpoint; if the API is unreachable it is a
SILENT no-op (exit 0, no output) so it can NEVER block or slow a session.

Wire it into a project's .claude/settings.json:

  {
    "hooks": {
      "SessionStart": [
        { "hooks": [ { "type": "command",
            "command": "/path/to/synapse/.venv/Scripts/python.exe /path/to/synapse/scripts/session_brief.py <project_id>" } ] }
      ]
    }
  }

Use ``python -m scripts.wire_project <project_id>`` to write this file automatically
with the correct paths for the current machine.

Claude Code injects a SessionStart hook's stdout into the session as context.

    python scripts/session_brief.py acme-jobs
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request

API = os.environ.get("SYNAPSE_API_URL", "http://127.0.0.1:8848").rstrip("/")
_MAX = 6  # items per section — keep the injected context tight (T2/T11)


def _section(title: str, items) -> list[str]:
    items = [i for i in (items or []) if i]
    if not items:
        return []
    out = [f"**{title}:**"]
    out += [f"- {i}" for i in items[:_MAX]]
    out.append("")
    return out


def main() -> int:
    if len(sys.argv) < 2 or not sys.argv[1].strip():
        return 0
    project_id = sys.argv[1].strip()

    try:
        req = urllib.request.Request(f"{API}/api/v1/brief/{project_id}",
                                     headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=3) as resp:
            b = json.load(resp)
    except Exception:
        return 0  # API down / not reachable → stay silent, never block the session

    lines = [f"## Synapse brief — {project_id}", ""]
    if b.get("project_summary"):
        lines += [b["project_summary"], ""]
    lines += _section("Key decisions", b.get("key_decisions"))
    lines += _section("Active conventions", b.get("active_conventions"))
    lines += _section("Relevant lessons", b.get("relevant_lessons"))
    lines += _section("Cross-project knowledge", b.get("cross_project_knowledge"))

    body = "\n".join(lines).strip()
    if len(lines) <= 2:  # nothing but the header → don't inject noise
        return 0

    # Emit the structured SessionStart form so the brief is added as context explicitly;
    # plain stdout would also be injected, but this is unambiguous across CC versions.
    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "SessionStart",
            "additionalContext": body,
        }
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
