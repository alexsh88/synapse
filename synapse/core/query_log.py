"""Query telemetry — what was actually asked, so a held-out eval set can be built from it.

The golden set is 52 cases the author wrote, and it has been used while tuning. That makes it a
regression gate, not a quality measurement: it cannot say whether retrieval generalises, only
whether it got worse at the cases someone already thought of. The fix is a held-out set built from
queries nobody chose, which means capturing real ones first — hence this module.

What it records, per read:

* the query text, **after** running it through the credential redactor
* which tool asked (recall / search / brief), the scopes, and the project
* the uuids and scores of the top results, which is what lets a later pass pool candidates
  across several retrieval strategies without re-running the engine
* latency, because the p95 nobody measures is the p95 nobody fixes

What it deliberately does not record: fact text. The uuids are enough to reconstruct anything a
later analysis needs from the graph itself, and copying prose into a second store means a second
place for a credential to survive a redaction bug.

Storage is a capped Redis list rather than a file. Roughly ten MCP processes read concurrently and
append-mode writes from that many writers interleave on Windows; ``LPUSH`` does not. It is
telemetry, not knowledge, so it does not go in the graph (R2 — the graph stores decisions, not
traffic).

Every function here fails open. A read must never break because its telemetry could not be written.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from synapse.core.redaction import redact

logger = logging.getLogger("synapse.query_log")

#: Redis key holding the capped list of query records, newest first.
QUERY_LOG_KEY = "synapse:querylog"

#: Roughly a year of personal use at a few dozen reads a day, and a few MB of Redis. The cap
#: exists so an unattended loop cannot fill the instance that also holds the brief cache.
MAX_ENTRIES = 20_000

#: Only the head of the result list matters for pooling, and storing 50 uuids per read would make
#: the log an order of magnitude larger for candidates no judge would ever be shown.
TOP_N_RESULTS = 10

_WARNED = False


def build_record(
    *,
    tool: str,
    query: str,
    scopes: list[str] | None,
    results: list[Any],
    latency_ms: float,
    project_id: str | None = None,
    at: str | None = None,
) -> dict[str, Any]:
    """Shape one log record. Pure — separated from the write so it is trivially testable.

    The query is redacted here rather than at the call site so there is exactly one path into the
    log and it cannot be bypassed by a new caller who forgets.
    """
    safe_query, kinds = redact(query or "")
    hits = []
    for r in (results or [])[:TOP_N_RESULTS]:
        uuid = getattr(r, "uuid", None)
        if uuid is None:
            continue
        score = getattr(r, "score", None)
        hits.append({"uuid": uuid, "score": round(float(score), 4) if score is not None else None})
    record: dict[str, Any] = {
        "at": at or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tool": tool,
        "query": safe_query,
        "project_id": project_id,
        "scopes": list(scopes) if scopes else None,
        "n_results": len(results or []),
        "top": hits,
        "latency_ms": round(latency_ms, 1),
    }
    # Kinds only, never values — the same rule the write path follows. A query that carried a
    # credential is worth knowing about; the credential is not worth keeping.
    if kinds:
        record["redacted_kinds"] = kinds
    return record


async def record(redis, **kwargs: Any) -> None:
    """Append one record. Never raises, never blocks a read on a telemetry failure."""
    global _WARNED
    if redis is None:
        return
    try:
        entry = build_record(**kwargs)
        await redis.lpush(QUERY_LOG_KEY, json.dumps(entry))
        await redis.ltrim(QUERY_LOG_KEY, 0, MAX_ENTRIES - 1)
    except Exception as exc:  # noqa: BLE001 — telemetry is never worth failing a read for
        if not _WARNED:
            # Once per process: a Redis outage would otherwise emit this on every single read.
            logger.warning("query log unavailable, continuing without it (%s)", str(exc)[:120])
            _WARNED = True


async def read_all(redis, limit: int = MAX_ENTRIES) -> list[dict[str, Any]]:
    """Return recorded queries, newest first. Used by the eval tooling, not the read path."""
    if redis is None:
        return []
    raw = await redis.lrange(QUERY_LOG_KEY, 0, limit - 1)
    out: list[dict[str, Any]] = []
    for item in raw:
        try:
            out.append(json.loads(item))
        except (TypeError, ValueError):
            # A malformed entry is one lost sample, not a reason to lose the rest of the log.
            continue
    return out
