#!/usr/bin/env python
"""UserPromptSubmit hook helper — auto-recall relevant knowledge for each prompt.

Companion to ``session_brief.py``. Where the brief front-loads a project's knowledge ONCE at
session start, this fires on every user prompt: it recalls the most relevant facts for what the
user just typed and injects them as context — so RECALL happens deterministically instead of
depending on the agent choosing to call ``synapse:recall``.

Standalone (stdlib only, no synapse imports) so it runs under any Python without PYTHONPATH fuss.
Hits the running Synapse API's ``/recall`` endpoint. The gate that keeps this quiet is the
retrieval engine's own similarity floor, applied SERVER-SIDE on the absolute cosine similarity
(``min_relevance``, default 0.72): anything the endpoint returns has already cleared it. The
``score`` the endpoint reports back is the *composite* rank score (relevance+recency+confidence+
connectivity), which legitimately runs low for older-but-relevant facts — so we do NOT re-gate on
it here (that would double-filter on the wrong scale). We simply take the top few the server
already floored. FAIL-SILENT by design:

  * nothing clears the server floor     -> endpoint returns []  -> no output (never inject noise)
  * prompt is trivial / a slash command -> skipped (no wasted API call)
  * API unreachable / any error         -> exit 0, no output (never block or slow a prompt)

So on a prompt with no genuinely-relevant memory, this is a zero-cost no-op; it only speaks up
when there is a real hit. The recall is scoped to this project (+ global) via the project_id arg.

Wire it into a project's .claude/settings.json (``python -m scripts.wire_project`` does this):

  {
    "hooks": {
      "UserPromptSubmit": [
        { "hooks": [ { "type": "command",
            "command": "<synapse>/.venv/bin/python <synapse>/scripts/prompt_recall.py <project_id>" } ] }
            # Windows: <synapse>/.venv/Scripts/python.exe — `wire_project` picks the right one.
      ]
    }
  }

Claude Code passes the prompt as JSON on stdin and injects this hook's ``additionalContext``.

    echo '{"prompt": "how do we rank recall results?"}' | python scripts/prompt_recall.py synapse

Tune without re-wiring via env:
    SYNAPSE_RECALL_MIN   optional EXTRA clamp on the returned composite score (default 0.0 = off;
                         the server-side cosine floor is the real gate). Raise it only to be
                         stricter than the engine's own floor.
    SYNAPSE_API_URL      API base (default http://127.0.0.1:8848)
"""

from __future__ import annotations

import json
import os
import sys
import urllib.parse
import urllib.request

API = os.environ.get("SYNAPSE_API_URL", "http://127.0.0.1:8848").rstrip("/")
# Optional extra clamp on the RETURNED composite score. Default 0.0 = off: the composite blends
# recency/confidence/connectivity, so it is NOT a pure relevance measure — see the note below.
MIN_SCORE = float(os.environ.get("SYNAPSE_RECALL_MIN", "0.0"))
# Clamp on the `relevance` COMPONENT, which since 2026-07-25 is absolute and comparable across
# queries (rescaled from the engine's cosine floor rather than min-max normalized within the
# result set — research §1.4). That change is what makes a client-side clamp meaningful at all;
# the earlier attempt to gate on the composite silenced every query and had to be disabled.
# Measured on the live graph: strong hits land 0.27-0.35, usable ones ~0.15, and near-junk that
# only just cleared the server floor sits at 0.01-0.05. 0.10 drops that tail without touching
# real hits. Facts whose relevance is unavailable are always kept.
MIN_RELEVANCE = float(os.environ.get("SYNAPSE_RECALL_MIN_RELEVANCE", "0.10"))
_MAX = 5           # facts injected — keep the context tight (T2/T11)
_LIMIT = 8         # asked of the API before the client-side floor/cap
_MIN_PROMPT = 8    # alnum chars below this → skip (embeds poorly, rarely clears the floor)

# Exact low-signal prompts that never warrant a recall (saves a pointless API round-trip).
_TRIVIAL = {"yes", "no", "go", "ok", "okay", "sure", "continue", "next", "stop",
            "thanks", "thank you", "done", "y", "n", "yep", "nope", "proceed"}


def _read_prompt() -> str:
    # Read the RAW BYTES and decode UTF-8 ourselves. Claude Code pipes UTF-8, but on Windows
    # sys.stdin decodes with the locale codec (cp1252) — a multi-byte prompt then raises
    # UnicodeDecodeError. sys.stdin.buffer sidesteps the locale entirely; "replace" never throws.
    try:
        raw = sys.stdin.buffer.read().decode("utf-8", "replace") if not sys.stdin.isatty() else ""
    except Exception:
        return ""
    if not raw.strip():
        return ""
    try:                                   # Claude Code sends a JSON envelope with "prompt".
        return str(json.loads(raw).get("prompt", "")).strip()
    except (json.JSONDecodeError, AttributeError):
        return raw.strip()                 # tolerate a raw-text prompt too


def _worth_recalling(prompt: str) -> bool:
    if not prompt or prompt.startswith("/"):          # empty or a slash command
        return False
    if prompt.lower() in _TRIVIAL:
        return False
    return sum(c.isalnum() for c in prompt) >= _MIN_PROMPT


def _run() -> int:
    if len(sys.argv) < 2 or not sys.argv[1].strip():
        return 0
    project_id = sys.argv[1].strip()

    prompt = _read_prompt()
    if not _worth_recalling(prompt):
        return 0

    # feedback=true: this injection is a real consumption, so it counts as an impression
    # (roadmap item 14). Eval runs and UI browsing deliberately leave it off.
    qs = urllib.parse.urlencode({"q": prompt[:2000], "project": project_id,
                                 "limit": _LIMIT, "feedback": "true"})
    try:
        req = urllib.request.Request(f"{API}/api/v1/recall?{qs}",
                                     headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=3) as resp:
            hits = json.load(resp)
    except Exception:
        return 0  # API down / not reachable → stay silent, never block the prompt

    facts: list[str] = []
    seen: set[str] = set()
    for h in hits if isinstance(hits, list) else []:
        fact = str(h.get("fact", "")).strip()
        if not fact or fact in seen:
            continue
        if MIN_SCORE and float(h.get("score", 0.0)) < MIN_SCORE:   # optional extra clamp (off by default)
            continue
        relevance = (h.get("components") or {}).get("relevance")
        if MIN_RELEVANCE and relevance is not None and float(relevance) < MIN_RELEVANCE:
            continue  # cleared the server floor but is not actually about this prompt
        seen.add(fact)
        # flag cross-project (global) hits — the surprising, high-value ones
        facts.append(f"- {fact}" + (" · _global_" if h.get("scope") == "global" else ""))
        if len(facts) >= _MAX:
            break

    if not facts:
        return 0  # nothing cleared the floor → inject nothing

    body = "\n".join([
        "## Synapse — possibly relevant prior knowledge (auto-recalled)",
        "_Surfaced for this prompt; verify before relying on it, and `synapse:update` if stale._",
        "",
        *facts,
    ])

    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": body,
        }
    }))
    return 0


def main() -> int:
    # This hook must NEVER surface a traceback to the user or block a prompt. _run() already
    # guards its network call; this outer net is the last line of the fail-silent contract, so
    # any unforeseen edge case (odd stdin, unexpected API shape, stdout encoding) stays silent.
    try:
        return _run()
    except Exception:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
