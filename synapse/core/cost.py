"""Cost accounting — what Synapse actually spends, recorded as it spends it.

The question "is this saving me more than it costs" was unanswerable for months, because the system
recorded none of its own spend. Every estimate had to be reconstructed afterwards from an Anthropic
invoice that does not distinguish Synapse's writes from anything else on the same key.

What is observable, and what is not
-----------------------------------
Synapse makes some LLM calls directly (triage, session capture) and hands the rest to Graphiti,
which owns its own clients. So:

* **Direct calls** — exact input/output tokens from the provider response, and therefore an exact
  cost. Recorded with ``record_usage``.
* **Graphiti-mediated extraction** — Graphiti does not surface usage back to the caller, so tokens
  are not available without forking it. What *is* available at ``HybridLLMClient`` is which
  provider served each call and for which extraction step, which is recorded with ``record_call``.

That split is stated rather than papered over. A ledger that silently guessed at the unobservable
half would be worse than one that reports "42 cloud extraction calls, tokens unknown" — the second
is a number you can act on, and it makes the local-vs-cloud mix visible, which is the decision the
``extraction_mode`` setting actually turns on.

Storage is a capped Redis list, same reasoning as ``query_log``: many processes, concurrent writes,
and this is telemetry rather than knowledge so it does not belong in the graph.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

logger = logging.getLogger("synapse.cost")

COST_LOG_KEY = "synapse:costlog"
MAX_ENTRIES = 50_000

#: USD per MILLION tokens, (input, output).
#:
#: These are defaults, not gospel — provider pricing changes and this table will drift. It is here
#: so the ledger reports dollars rather than raw tokens, and it is overridable per deployment.
#: Verify against current published pricing before quoting a figure from this ledger anywhere that
#: matters. An unknown model costs 0.0 and is counted, never silently dropped: a missing price is a
#: gap in the table, not evidence of a free call.
PRICES: dict[str, tuple[float, float]] = {
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-sonnet-4-5": (3.00, 15.00),
    "claude-opus-5": (15.00, 75.00),
    "claude-haiku-4-5": (1.00, 5.00),
    # Local models are free at the margin — the GPU is already paid for. Recorded anyway so the
    # ledger can show what the local path WOULD have cost if it had gone to the cloud.
    "gemma3:12b": (0.0, 0.0),
    "bge-m3": (0.0, 0.0),
}

_WARNED = False


def price_of(model: str) -> tuple[float, float]:
    """(input, output) USD per million tokens. Prefix match, so dated snapshots resolve."""
    if model in PRICES:
        return PRICES[model]
    for known, price in PRICES.items():
        if model.startswith(known):
            return price
    return (0.0, 0.0)


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """Dollar cost of one call. Rounded to 6 dp — per-call costs are genuinely that small."""
    in_price, out_price = price_of(model)
    return round(
        (input_tokens / 1_000_000) * in_price + (output_tokens / 1_000_000) * out_price, 6
    )


def build_entry(
    *,
    operation: str,
    model: str,
    provider: str,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    at: str | None = None,
) -> dict[str, Any]:
    """One ledger line. Tokens may be None — that is the Graphiti-mediated case, not an error."""
    entry: dict[str, Any] = {
        "at": at or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "operation": operation,
        "model": model,
        "provider": provider,
    }
    if input_tokens is not None or output_tokens is not None:
        i, o = input_tokens or 0, output_tokens or 0
        entry["input_tokens"] = i
        entry["output_tokens"] = o
        entry["cost_usd"] = cost_usd(model, i, o)
        entry["priced"] = model in PRICES or any(model.startswith(k) for k in PRICES)
    else:
        # Explicit, so a reader can tell "this call was free" from "we could not see the tokens".
        entry["tokens_unavailable"] = True
    return entry


async def _append(redis, entry: dict[str, Any]) -> None:
    global _WARNED
    if redis is None:
        return
    try:
        await redis.lpush(COST_LOG_KEY, json.dumps(entry))
        await redis.ltrim(COST_LOG_KEY, 0, MAX_ENTRIES - 1)
    except Exception as exc:  # noqa: BLE001 — accounting never fails the work it is accounting for
        if not _WARNED:
            logger.warning("cost ledger unavailable, continuing without it (%s)", str(exc)[:120])
            _WARNED = True


async def record_usage(
    redis, *, operation: str, model: str, provider: str,
    input_tokens: int, output_tokens: int,
) -> None:
    """Record a call whose token usage the provider gave us. Exact cost."""
    await _append(redis, build_entry(
        operation=operation, model=model, provider=provider,
        input_tokens=input_tokens, output_tokens=output_tokens,
    ))


async def record_call(redis, *, operation: str, model: str, provider: str) -> None:
    """Record a call we can attribute but not price — Graphiti-mediated extraction."""
    await _append(redis, build_entry(operation=operation, model=model, provider=provider))


def summarize(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate a ledger into the shape a human actually asks for.

    Reports priced and unpriced calls separately on purpose. Collapsing them would let 400 cloud
    extraction calls with unknown tokens hide inside a confident-looking dollar total.
    """
    total_cost = 0.0
    priced_calls = 0
    unpriced_calls = 0
    by_operation: dict[str, dict[str, Any]] = {}
    by_provider: dict[str, int] = {}
    by_day: dict[str, dict[str, Any]] = {}

    for e in entries:
        op = e.get("operation", "unknown")
        provider = e.get("provider", "unknown")
        by_provider[provider] = by_provider.get(provider, 0) + 1
        slot = by_operation.setdefault(op, {"calls": 0, "cost_usd": 0.0, "unpriced": 0})
        slot["calls"] += 1

        # Days carry call counts as well as cost. Accumulating only cost would drop a day whose
        # calls were all unpriced out of the report entirely, which reads as "nothing happened"
        # when what actually happened was activity we could not price — the opposite of the point.
        day = str(e.get("at", ""))[:10]
        day_slot = by_day.setdefault(day, {"cost_usd": 0.0, "calls": 0, "unpriced": 0}) if day else None

        if day_slot is not None:
            day_slot["calls"] += 1

        if e.get("tokens_unavailable"):
            unpriced_calls += 1
            slot["unpriced"] += 1
            if day_slot is not None:
                day_slot["unpriced"] += 1
            continue

        priced_calls += 1
        c = float(e.get("cost_usd") or 0.0)
        total_cost += c
        slot["cost_usd"] = round(slot["cost_usd"] + c, 6)
        if day_slot is not None:
            day_slot["cost_usd"] = round(day_slot["cost_usd"] + c, 6)

    return {
        "calls": len(entries),
        "priced_calls": priced_calls,
        "unpriced_calls": unpriced_calls,
        "total_cost_usd": round(total_cost, 4),
        "by_operation": by_operation,
        "by_provider": by_provider,
        "by_day": dict(sorted(by_day.items(), reverse=True)),
    }


_shared: Any = None
_shared_tried = False


def shared_client() -> Any:
    """A lazily-built Redis client for callers that have no handle to thread through.

    The LLM calls worth accounting for happen in module-level helpers and inside Graphiti's client
    stack, neither of which is given an engine. Plumbing a client to both would mean changing
    signatures across the write path to carry a telemetry dependency, so the dependency is resolved
    here instead. Built once per process; a failure to build is remembered, not retried per call.
    """
    global _shared, _shared_tried
    if _shared_tried:
        return _shared
    _shared_tried = True
    from synapse.config import settings

    if not settings.redis_url:
        return None
    try:
        import redis.asyncio as aioredis

        _shared = aioredis.from_url(settings.redis_url, decode_responses=True)
    except Exception as exc:  # noqa: BLE001
        logger.warning("cost ledger has no Redis (%s)", str(exc)[:120])
        _shared = None
    return _shared


async def track_usage(**kwargs: Any) -> None:
    """``record_usage`` against the shared client, for callers with no handle."""
    await record_usage(shared_client(), **kwargs)


async def track_call(**kwargs: Any) -> None:
    """``record_call`` against the shared client, for callers with no handle."""
    await record_call(shared_client(), **kwargs)


async def read_all(redis, limit: int = MAX_ENTRIES) -> list[dict[str, Any]]:
    if redis is None:
        return []
    raw = await redis.lrange(COST_LOG_KEY, 0, limit - 1)
    out: list[dict[str, Any]] = []
    for item in raw:
        try:
            out.append(json.loads(item))
        except (TypeError, ValueError):
            continue
    return out
