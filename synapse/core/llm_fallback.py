"""Credit-aware JSON LLM helper — use Anthropic (Haiku) when the key has credit, else local gemma.

Triage and the capture judge are cheap Haiku JSON calls. This wrapper tries Haiku first; on a
credit/auth failure it marks Anthropic unavailable for a cooldown and transparently falls back to the
local Ollama model — so Synapse keeps working when credits run out. Callers learn which provider ran
(`"haiku"` vs `"local"`) so they can be more conservative on the weaker local path (e.g. route captures
to the review queue). Returns RAW text; callers parse leniently (the prompts ask for JSON).

**Shared cooldown (WP-B item 5):** ~10 MCP server processes run in parallel. Without coordination each
one independently rediscovers credit exhaustion and holds its own 5-minute cooldown — so the first
Haiku call in every process is guaranteed to fail. The cooldown is therefore mirrored into Redis under
a single key with a TTL == the cooldown window: the first process to see exhaustion marks it, and the
others read it and skip Haiku until the TTL lapses. Redis is best-effort only — the in-process state is
always authoritative enough that a Redis outage can never break the fallback logic itself.
"""

from __future__ import annotations

import logging
import time

from synapse.config import settings

logger = logging.getLogger("synapse.llm_fallback")

_COOLDOWN_S = 300.0  # after a credit failure, don't retry Anthropic for 5 min

# Redis key shared across all MCP processes. Presence => Anthropic is in cooldown; the key's TTL is the
# remaining cooldown, so it self-expires and lets the next call retry Haiku without any explicit reset.
_REDIS_COOLDOWN_KEY = "synapse:anthropic:cooldown"

# Process-local Redis client, lazily built and cached. `False` means "we tried and Redis is unreachable"
# so we stop retrying the connection on every call (a Redis outage stays cheap). `None` means "not yet
# attempted". Any real client is truthy.
_redis_client: object | None = None
_redis_broken: bool = False


class _AnthropicState:
    available: bool = True
    retry_at: float = 0.0


_state = _AnthropicState()


def _now() -> float:
    return time.monotonic()


def _get_redis():
    """Best-effort shared Redis client. Returns None if Redis is unavailable — never raises.

    Uses the sync `redis` client (the fallback helpers run inside async code but the get/setex calls are
    sub-millisecond and this keeps a Redis outage from ever propagating an exception into the LLM path).
    """
    global _redis_client, _redis_broken
    if _redis_broken:
        return None
    if _redis_client is not None:
        return _redis_client
    try:
        import redis  # lazy: optional dependency, only touched when the shared path is exercised

        client = redis.Redis.from_url(
            settings.redis_url, socket_connect_timeout=0.25, socket_timeout=0.25
        )
        _redis_client = client
        return client
    except Exception as exc:  # noqa: BLE001 — Redis is best-effort; degrade to in-process state
        logger.debug("Redis unavailable for shared fallback state (%s); using in-process only", exc)
        _redis_broken = True
        return None


def _redis_cooldown_active() -> bool | None:
    """True/False from Redis, or None if Redis can't answer (caller falls back to in-process state)."""
    client = _get_redis()
    if client is None:
        return None
    try:
        return bool(client.exists(_REDIS_COOLDOWN_KEY))
    except Exception as exc:  # noqa: BLE001 — a Redis blip must not break availability checks
        logger.debug("Redis exists() failed (%s); using in-process state", exc)
        return None


def _redis_set_cooldown() -> None:
    """Publish the cooldown to Redis with a TTL == the cooldown window. Best-effort; never raises."""
    client = _get_redis()
    if client is None:
        return
    try:
        client.set(_REDIS_COOLDOWN_KEY, "1", ex=int(_COOLDOWN_S))
    except Exception as exc:  # noqa: BLE001
        logger.debug("Redis setex() failed (%s); cooldown stays process-local", exc)


def _redis_clear_cooldown() -> None:
    """Clear the shared cooldown (used by reset_state so tests/ops start clean). Best-effort."""
    client = _get_redis()
    if client is None:
        return
    try:
        client.delete(_REDIS_COOLDOWN_KEY)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Redis delete() failed (%s)", exc)


def is_credit_error(exc: Exception) -> bool:
    s = str(exc).lower()
    return (
        "credit balance" in s or "billing" in s or "quota" in s or "insufficient" in s
        or getattr(exc, "status_code", None) in (401, 403)
        or "authentication_error" in s
    )


def anthropic_available() -> bool:
    """True unless a credit/auth failure is in cooldown (in this process OR any peer process via Redis).

    The shared Redis flag lets ~10 MCP processes learn about exhaustion from whichever hit it first,
    instead of each paying its own first-call failure. Redis is advisory: if it's unreachable we rely on
    the in-process cooldown, so a Redis outage never breaks the fallback.
    """
    if not settings.anthropic_api_key:
        return False

    # Local monotonic cooldown is authoritative for THIS process's own recovery: once it elapses, we
    # allow a retry and proactively clear the shared flag so peers recover too (Redis TTL is wall-clock;
    # our own elapsed cooldown is the more precise signal that it's worth retrying Haiku).
    local_in_cooldown = not _state.available and _now() < _state.retry_at
    if not _state.available and _now() >= _state.retry_at:
        _state.available = True
        _state.retry_at = 0.0
        _redis_clear_cooldown()

    if local_in_cooldown:
        return False

    # Not in local cooldown → consult the shared flag: a PEER may have discovered exhaustion. Mirror it
    # locally so behaviour is consistent for code paths that read _state directly.
    shared = _redis_cooldown_active()
    if shared is True:
        _state.available = False
        _state.retry_at = max(_state.retry_at, _now() + _COOLDOWN_S)
        return False

    return _state.available


def _mark_unavailable() -> None:
    _state.available = False
    _state.retry_at = _now() + _COOLDOWN_S
    _redis_set_cooldown()
    logger.warning("Anthropic unavailable (credit/auth) — falling back to local for %.0fs", _COOLDOWN_S)


def reset_state() -> None:
    """Test/ops helper — clears both the in-process and shared (Redis) cooldown."""
    _state.available = True
    _state.retry_at = 0.0
    _redis_clear_cooldown()


async def haiku_or_local(system: str, user: str, *, max_tokens: int = 800) -> tuple[str, str]:
    """(raw_text, provider). Tries Haiku when credits allow; falls back to local gemma otherwise."""
    if anthropic_available():
        try:
            from anthropic import AsyncAnthropic

            client = AsyncAnthropic(api_key=settings.anthropic_api_key)
            msg = await client.messages.create(
                model=settings.triage_model, max_tokens=max_tokens,
                system=system, messages=[{"role": "user", "content": user}])
            text = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text").strip()
            return text, "haiku"
        except Exception as exc:  # noqa: BLE001
            if is_credit_error(exc):
                _mark_unavailable()
            else:
                logger.warning("Haiku call failed (%s); trying local", str(exc)[:100])

    from openai import AsyncOpenAI

    client = AsyncOpenAI(base_url=settings.ollama_base_url, api_key="ollama")
    resp = await client.chat.completions.create(
        model=settings.local_extraction_model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0, max_tokens=max_tokens)
    return resp.choices[0].message.content or "", "local"
