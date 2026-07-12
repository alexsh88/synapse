"""Extraction routing — hybrid fallback + mode factory."""

from __future__ import annotations

from synapse.core.extraction_clients import (
    HybridLLMClient, OllamaStrictClient, build_extraction_client,
)


class _Boom:
    async def generate_response(self, *a, **k):
        raise ValueError("malformed JSON from local model")


class _Records:
    def __init__(self, ret):
        self.ret, self.called = ret, False

    async def generate_response(self, *a, **k):
        self.called = True
        return self.ret


async def test_hybrid_falls_back_to_cloud_on_local_failure(monkeypatch):
    # Isolate from the global credit-availability gate: it consults process/Redis
    # state that other tests (or a live stack) may have flipped. This test covers
    # the fallback wiring, not the credit gate.
    from synapse.core import llm_fallback

    monkeypatch.setattr(llm_fallback, "anthropic_available", lambda: True)
    cloud = _Records({"src": "cloud"})
    out = await HybridLLMClient(local=_Boom(), cloud=cloud).generate_response([], response_model=None)
    assert out == {"src": "cloud"} and cloud.called


async def test_hybrid_uses_local_when_it_succeeds():
    cloud = _Records({"src": "cloud"})
    local = _Records({"src": "local"})
    out = await HybridLLMClient(local=local, cloud=cloud).generate_response([])
    assert out == {"src": "local"} and not cloud.called   # cloud untouched


def test_factory_selects_client_by_mode(monkeypatch):
    from synapse.core import extraction_clients as ec
    monkeypatch.setattr(ec, "build_cloud_client", lambda: object())   # avoid needing an API key

    monkeypatch.setattr(ec.settings, "extraction_mode", "hybrid")
    assert isinstance(ec.build_extraction_client(), HybridLLMClient)

    monkeypatch.setattr(ec.settings, "extraction_mode", "local")
    assert isinstance(ec.build_extraction_client(), OllamaStrictClient)

    monkeypatch.setattr(ec.settings, "extraction_mode", "cloud")
    assert build_extraction_client() is not None                      # the cloud stub
