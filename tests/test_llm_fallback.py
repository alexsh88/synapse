"""Credit-aware Anthropic→local fallback state machine."""

from __future__ import annotations

from synapse.core import llm_fallback as lf


def test_is_credit_error_detects_billing():
    assert lf.is_credit_error(Exception("Your credit balance is too low to access the Anthropic API"))
    assert lf.is_credit_error(Exception("billing required"))
    assert not lf.is_credit_error(Exception("connection timed out"))


def test_availability_marks_down_then_recovers_after_cooldown(monkeypatch):
    lf.reset_state()
    monkeypatch.setattr(lf.settings, "anthropic_api_key", "sk-test")
    assert lf.anthropic_available()

    lf._mark_unavailable()
    assert not lf.anthropic_available()                       # in cooldown

    monkeypatch.setattr(lf, "_now", lambda: lf._state.retry_at + 1)  # cooldown elapsed
    assert lf.anthropic_available()                           # auto-recovers
    lf.reset_state()


def test_unavailable_without_a_key(monkeypatch):
    lf.reset_state()
    monkeypatch.setattr(lf.settings, "anthropic_api_key", "")
    assert not lf.anthropic_available()
    lf.reset_state()


# --- WP-B item 5: Redis-shared cooldown state -------------------------------


class _FakeRedis:
    """Minimal in-memory Redis stand-in shared across simulated processes (no TTL expiry needed here)."""

    def __init__(self):
        self.store: dict[str, str] = {}

    def exists(self, key):
        return 1 if key in self.store else 0

    def set(self, key, val, ex=None):
        self.store[key] = val

    def delete(self, key):
        self.store.pop(key, None)


def test_fallback_state_shared_via_redis(monkeypatch):
    # Two MCP processes share ONE Redis. Process A discovers credit exhaustion (marks the shared key);
    # process B — with fresh in-process state — must learn it's unavailable WITHOUT its own failed call.
    fake = _FakeRedis()
    monkeypatch.setattr(lf.settings, "anthropic_api_key", "sk-test")
    monkeypatch.setattr(lf, "_get_redis", lambda: fake)
    lf.reset_state()

    # Process A hits exhaustion.
    lf._mark_unavailable()
    assert fake.store.get(lf._REDIS_COOLDOWN_KEY) == "1"   # published to Redis
    assert not lf.anthropic_available()                    # A is down

    # Process B: simulate a *separate* process by resetting only the in-process state (Redis persists).
    lf._state.available = True
    lf._state.retry_at = 0.0
    assert not lf.anthropic_available()                    # B learns it from Redis — no own failure

    # When the shared key clears (TTL lapsed / cleared), B recovers.
    fake.delete(lf._REDIS_COOLDOWN_KEY)
    lf._state.available = True
    lf._state.retry_at = 0.0
    assert lf.anthropic_available()
    lf.reset_state()


def test_fallback_survives_redis_down(monkeypatch):
    # A Redis outage must never break the fallback: availability falls back to in-process cooldown.
    monkeypatch.setattr(lf.settings, "anthropic_api_key", "sk-test")
    monkeypatch.setattr(lf, "_get_redis", lambda: None)   # Redis unreachable
    lf.reset_state()

    assert lf.anthropic_available()          # no Redis, no local cooldown → available
    lf._mark_unavailable()                   # _redis_set_cooldown is a no-op; local cooldown still set
    assert not lf.anthropic_available()      # in-process cooldown holds

    monkeypatch.setattr(lf, "_now", lambda: lf._state.retry_at + 1)  # local cooldown elapses
    assert lf.anthropic_available()          # recovers purely from in-process state
    lf.reset_state()


def test_get_redis_never_raises_on_bad_url(monkeypatch):
    # A malformed/unreachable URL must degrade to None, not raise.
    monkeypatch.setattr(lf, "_redis_client", None)
    monkeypatch.setattr(lf, "_redis_broken", False)

    def _boom(*a, **k):
        raise RuntimeError("cannot connect")

    import redis as _redis_mod
    monkeypatch.setattr(_redis_mod.Redis, "from_url", staticmethod(_boom))
    assert lf._get_redis() is None
    # And subsequent calls stay cheap (cached as broken).
    assert lf._get_redis() is None
