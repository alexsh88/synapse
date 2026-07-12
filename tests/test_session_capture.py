"""CaptureEngine — confidence routing + the three R2 gates. No live services."""

from __future__ import annotations

from types import SimpleNamespace

from synapse.core.session_capture import CaptureCandidate, CaptureEngine

TRANSCRIPT = "user: " + ("a real coding conversation. " * 30)  # > 200 chars


class FakeDriver:
    def __init__(self):
        self.queries: list[str] = []

    async def execute_query(self, query, **params):
        self.queries.append(query)
        return SimpleNamespace(records=[])


class _FG:
    def __init__(self, driver):
        self.driver = driver


def _remember(outcome="stored"):
    calls = []

    async def remember(content, knowledge_type=None, project_id=None, source=None, force=False):
        calls.append({"content": content, "type": knowledge_type, "project": project_id, "force": force})
        return SimpleNamespace(outcome=SimpleNamespace(value=outcome))

    return remember, calls


def _judge(candidates, provider="haiku"):
    async def judge(_transcript):
        return candidates, provider
    return judge


def _engine(driver, judge, remember, **kw):
    kw.setdefault("autostore_threshold", 0.8)
    kw.setdefault("enabled", True)
    return CaptureEngine(_FG(driver), judge, remember, **kw)


def _queued(driver) -> bool:
    return any("PendingCapture" in q and "MERGE" in q for q in driver.queries)


async def test_high_confidence_durable_autostores():
    d = FakeDriver()
    rem, calls = _remember("stored")
    e = _engine(d, _judge([CaptureCandidate(content="Acme-API uses BigDecimal for money.",
                                            type="convention", confidence=0.9)]), rem)
    r = await e.capture("acme-api", "s1", TRANSCRIPT)
    assert r.stored == ["Acme-API uses BigDecimal for money."] and not r.pending
    assert calls[0]["type"] == "convention" and calls[0]["project"] == "acme-api"
    assert not _queued(d)


async def test_low_confidence_queues_for_review():
    d = FakeDriver()
    rem, calls = _remember()
    e = _engine(d, _judge([CaptureCandidate(content="might matter", type="lesson", confidence=0.5)]), rem)
    r = await e.capture("acme-api", "s1", TRANSCRIPT)
    assert r.pending == ["might matter"] and not r.stored and not calls
    assert _queued(d)


async def test_nondurable_type_queues_even_if_confident():
    d = FakeDriver()
    rem, calls = _remember()
    e = _engine(d, _judge([CaptureCandidate(content="some entity", type="entity", confidence=0.95)]), rem)
    r = await e.capture("p", "s", TRANSCRIPT)
    assert r.pending and not r.stored and not calls


async def test_pipeline_rejection_is_gate_three():
    # high-conf durable goes to remember(), but the write pipeline rejects it → not counted as stored.
    d = FakeDriver()
    rem, calls = _remember("rejected")
    e = _engine(d, _judge([CaptureCandidate(content="noise", type="lesson", confidence=0.9)]), rem)
    r = await e.capture("p", "s", TRANSCRIPT)
    assert not r.stored and not r.pending and calls  # called, but dropped


async def test_short_transcript_skipped():
    d = FakeDriver()
    rem, calls = _remember()
    e = _engine(d, _judge([CaptureCandidate(content="x", type="lesson", confidence=0.9)]), rem)
    r = await e.capture("p", "s", "too short")
    assert r.skipped and not calls


async def test_disabled_skips():
    d = FakeDriver()
    rem, _ = _remember()
    r = await _engine(d, _judge([]), rem, enabled=False).capture("p", "s", TRANSCRIPT)
    assert r.skipped


async def test_empty_judge_yields_nothing():
    d = FakeDriver()
    rem, calls = _remember()
    r = await _engine(d, _judge([]), rem).capture("p", "s", TRANSCRIPT)
    assert not r.stored and not r.pending and not r.skipped and not calls


async def test_degraded_local_judge_queues_everything_for_approval():
    # credits out -> judge ran on local gemma -> even a high-confidence durable item is QUEUED, not stored.
    d = FakeDriver()
    rem, calls = _remember("stored")
    e = _engine(d, _judge([CaptureCandidate(content="a durable rule", type="convention", confidence=0.95)],
                          provider="local"), rem)
    r = await e.capture("acme-api", "s1", TRANSCRIPT)
    assert r.pending == ["a durable rule"] and not r.stored and not calls
    assert _queued(d)
