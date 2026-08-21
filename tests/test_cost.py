"""Cost ledger — pricing, the priced/unpriced split, and fail-open recording."""

from __future__ import annotations

import json

from synapse.core import cost
from synapse.core.cost import (
    COST_LOG_KEY,
    MAX_ENTRIES,
    build_entry,
    cost_usd,
    price_of,
    read_all,
    record_call,
    record_usage,
    summarize,
)


class FakeRedis:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.lists: dict[str, list[str]] = {}
        self.trims: list[tuple[str, int, int]] = []

    async def lpush(self, key, value):
        if self.fail:
            raise ConnectionError("redis down")
        self.lists.setdefault(key, []).insert(0, value)

    async def ltrim(self, key, start, stop):
        self.trims.append((key, start, stop))

    async def lrange(self, key, start, stop):
        return self.lists.get(key, [])[start:stop + 1]


# ── pricing ───────────────────────────────────────────────────────────────────

def test_a_dated_model_snapshot_resolves_by_prefix():
    """Provider model ids carry dates; the table should not need a row per snapshot."""
    assert price_of("claude-haiku-4-5-20251001") == price_of("claude-haiku-4-5")


def test_an_unknown_model_prices_at_zero_rather_than_guessing():
    assert price_of("some-model-we-have-never-seen") == (0.0, 0.0)


def test_cost_is_computed_per_million_tokens():
    # 1M in + 1M out on a (3.00, 15.00) model.
    assert cost_usd("claude-sonnet-4-6", 1_000_000, 1_000_000) == 18.0


def test_small_calls_keep_meaningful_precision():
    """Per-call costs are genuinely tiny; rounding them to cents would report every call as free."""
    assert cost_usd("claude-haiku-4-5", 1200, 300) > 0.0


def test_local_models_cost_nothing_but_are_still_priced_entries():
    assert cost_usd("gemma3:12b", 500_000, 500_000) == 0.0


# ── entries ───────────────────────────────────────────────────────────────────

def test_a_priced_entry_carries_tokens_and_cost():
    e = build_entry(operation="triage", model="claude-haiku-4-5", provider="anthropic",
                    input_tokens=1000, output_tokens=200)
    assert e["input_tokens"] == 1000 and e["output_tokens"] == 200
    assert e["cost_usd"] > 0 and e["priced"] is True
    assert "tokens_unavailable" not in e


def test_an_unpriced_entry_says_so_explicitly():
    """"We could not see the tokens" must be distinguishable from "this call was free"."""
    e = build_entry(operation="extract:ExtractedEdges", model="claude-sonnet-4-6",
                    provider="anthropic")
    assert e["tokens_unavailable"] is True
    assert "cost_usd" not in e


def test_an_unknown_model_is_flagged_as_unpriced_not_silently_free():
    e = build_entry(operation="triage", model="mystery-model", provider="anthropic",
                    input_tokens=10_000, output_tokens=10_000)
    assert e["priced"] is False
    assert e["cost_usd"] == 0.0, "a missing price is a gap in the table, not a free call"


# ── recording ─────────────────────────────────────────────────────────────────

async def test_record_usage_appends_and_caps():
    redis = FakeRedis()
    await record_usage(redis, operation="triage", model="claude-haiku-4-5",
                       provider="anthropic", input_tokens=100, output_tokens=10)
    assert len(redis.lists[COST_LOG_KEY]) == 1
    assert redis.trims == [(COST_LOG_KEY, 0, MAX_ENTRIES - 1)]


async def test_record_call_appends_an_unpriced_entry():
    redis = FakeRedis()
    await record_call(redis, operation="extract:ExtractedEntities", model="gemma3:12b",
                      provider="ollama")
    stored = json.loads(redis.lists[COST_LOG_KEY][0])
    assert stored["tokens_unavailable"] is True


async def test_accounting_never_fails_the_work_it_accounts_for(monkeypatch):
    monkeypatch.setattr(cost, "_WARNED", False)
    await record_usage(FakeRedis(fail=True), operation="triage", model="m", provider="p",
                       input_tokens=1, output_tokens=1)


async def test_recording_without_redis_is_a_no_op():
    await record_usage(None, operation="triage", model="m", provider="p",
                       input_tokens=1, output_tokens=1)
    assert await read_all(None) == []


# ── summary ───────────────────────────────────────────────────────────────────

def _entries():
    return [
        build_entry(operation="triage", model="claude-haiku-4-5", provider="anthropic",
                    input_tokens=1_000_000, output_tokens=0, at="2026-08-19T10:00:00Z"),
        build_entry(operation="triage", model="gemma3:12b", provider="ollama",
                    input_tokens=1_000_000, output_tokens=0, at="2026-08-19T11:00:00Z"),
        build_entry(operation="extract:ExtractedEdges", model="claude-sonnet-4-6",
                    provider="anthropic", at="2026-08-18T09:00:00Z"),
        build_entry(operation="extract:ExtractedEdges", model="gemma3:12b",
                    provider="ollama", at="2026-08-18T09:05:00Z"),
    ]


def test_summary_separates_priced_from_unpriced_calls():
    """Collapsing them would let unpriced cloud calls hide inside a confident dollar total."""
    s = summarize(_entries())
    assert s["calls"] == 4 and s["priced_calls"] == 2 and s["unpriced_calls"] == 2


def test_summary_totals_only_what_it_can_price():
    s = summarize(_entries())
    assert s["total_cost_usd"] == 1.0  # 1M Haiku input at $1.00/M; gemma is free


def test_summary_breaks_down_by_operation_and_counts_unpriced_separately():
    s = summarize(_entries())
    assert s["by_operation"]["triage"]["calls"] == 2
    assert s["by_operation"]["extract:ExtractedEdges"]["unpriced"] == 2
    assert s["by_operation"]["extract:ExtractedEdges"]["cost_usd"] == 0.0


def test_summary_shows_the_provider_mix_which_is_what_extraction_mode_turns_on():
    s = summarize(_entries())
    assert s["by_provider"] == {"anthropic": 2, "ollama": 2}


def test_summary_reports_spend_by_day_newest_first():
    s = summarize(_entries())
    assert list(s["by_day"]) == ["2026-08-19", "2026-08-18"]
    assert s["by_day"]["2026-08-19"]["cost_usd"] == 1.0


def test_a_day_of_only_unpriced_calls_still_appears():
    """Dropping it would read as "nothing happened" when the truth is "we could not price it"."""
    s = summarize(_entries())
    assert s["by_day"]["2026-08-18"] == {"cost_usd": 0.0, "calls": 2, "unpriced": 2}


def test_summary_of_an_empty_ledger_is_zeroed_not_an_error():
    s = summarize([])
    assert s["calls"] == 0 and s["total_cost_usd"] == 0.0 and s["by_day"] == {}


async def test_read_all_skips_malformed_entries():
    redis = FakeRedis()
    await record_call(redis, operation="op", model="m", provider="p")
    redis.lists[COST_LOG_KEY].append("{broken")
    assert len(await read_all(redis)) == 1
