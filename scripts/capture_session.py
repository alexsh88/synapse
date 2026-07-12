#!/usr/bin/env python
"""PreCompact / SessionEnd hook — capture durable lessons from a session into Synapse.

Wire both events (in a project's .claude/settings.json) to:
    <python> C:/.../synapse/scripts/capture_session.py <project_id>
Claude Code passes `transcript_path` + `session_id` on stdin. This reads the NEW transcript turns
since the last run (a `<transcript>.synapse-offset` marker), and POSTs them to /api/v1/capture, where
a Haiku judge extracts durable knowledge (auto-store high-confidence, queue the rest for review).

Stdlib only; SILENT no-op on any error or if the API is down (exit 0) — it must never block a session.
The marker only advances after a successful POST, so nothing is lost if the API is offline.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request

API = os.environ.get("SYNAPSE_API_URL", "http://127.0.0.1:8848").rstrip("/")
MIN_CHARS = 200


def _turn(line: str) -> str | None:
    try:
        obj = json.loads(line)
    except Exception:
        return None
    msg = obj.get("message") if isinstance(obj.get("message"), dict) else obj
    role = msg.get("role") or obj.get("type")
    if role not in ("user", "assistant"):
        return None
    content = msg.get("content")
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        text = " ".join(b.get("text", "") for b in content
                        if isinstance(b, dict) and b.get("type") == "text")
    else:
        return None
    text = text.strip()
    return f"{role}: {text}" if text else None


def main() -> int:
    if len(sys.argv) < 2 or not sys.argv[1].strip():
        return 0
    project_id = sys.argv[1].strip()
    try:
        hook = json.load(sys.stdin)
    except Exception:
        return 0
    tpath, sid = hook.get("transcript_path"), hook.get("session_id", "")
    if not tpath or not os.path.exists(tpath):
        return 0
    try:
        with open(tpath, encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
    except Exception:
        return 0

    marker = tpath + ".synapse-offset"
    offset = 0
    try:
        if os.path.exists(marker):
            offset = int((open(marker, encoding="utf-8").read().strip() or "0"))
    except Exception:
        offset = 0

    turns = [t for t in (_turn(line) for line in lines[offset:]) if t]
    transcript = "\n".join(turns)[:24000]

    def advance():
        try:
            with open(marker, "w", encoding="utf-8") as f:
                f.write(str(len(lines)))
        except Exception:
            pass

    if len(transcript) < MIN_CHARS:
        advance()                      # nothing durable in these turns; don't reprocess
        return 0

    body = json.dumps({"project_id": project_id, "session_id": sid, "transcript": transcript}).encode("utf-8")
    try:
        req = urllib.request.Request(f"{API}/api/v1/capture", data=body,
                                     headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=15)   # server backgrounds the heavy work; returns fast
        advance()                                  # only advance on success — else retry next hook
    except Exception:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
