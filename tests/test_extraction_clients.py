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


# --- empty local extraction is a FAILURE, not an answer -------------------------------
#
# Strict json_schema decoding guarantees the local model emits VALID json, not USEFUL json. On dense
# input gemma returns a well-formed but empty extraction, nothing raises, and before 2026-07-27 that
# empty result was returned as though it were an answer — storing the write with 0 facts.

# The gate reads `response_model.__name__`, so these stubs are NAMED, not annotated: assigning
# `__name__` in a class body is shadowed by the `type.__name__` data descriptor on the metaclass.
class ExtractedEntities:
    pass


class ExtractedEdges:
    pass


class NodeResolutions:
    """A dedupe model — empty here is the correct, common answer and must NOT escalate."""


def _hybrid(local_ret, cloud_ret, monkeypatch, *, credits=True):
    from synapse.core import llm_fallback
    monkeypatch.setattr(llm_fallback, "anthropic_available", lambda: credits)
    local, cloud = _Records(local_ret), _Records(cloud_ret)
    return HybridLLMClient(local=local, cloud=cloud), local, cloud


async def test_empty_node_extraction_escalates_to_cloud(monkeypatch):
    client, _, cloud = _hybrid({"extracted_entities": []}, {"extracted_entities": [{"name": "x"}]},
                               monkeypatch)
    out = await client.generate_response([], response_model=ExtractedEntities)
    assert cloud.called and out["extracted_entities"], "an empty local extraction must escalate"


async def test_empty_edge_extraction_escalates_to_cloud(monkeypatch):
    client, _, cloud = _hybrid({"edges": []}, {"edges": [{"fact": "x"}]}, monkeypatch)
    await client.generate_response([], response_model=ExtractedEdges)
    assert cloud.called


async def test_a_non_empty_local_extraction_is_kept(monkeypatch):
    client, _, cloud = _hybrid({"edges": [{"fact": "real"}]}, {"edges": []}, monkeypatch)
    out = await client.generate_response([], response_model=ExtractedEdges)
    assert out["edges"][0]["fact"] == "real" and not cloud.called   # no needless spend


async def test_an_empty_dedupe_answer_does_not_escalate(monkeypatch):
    """THE COST PIN. Dedupe/invalidation prompts answer "nothing" constantly and are CORRECT to.

    Escalating those would buy a Sonnet call on nearly every write to confirm an empty answer, which
    is why the gate is on the two extraction response models rather than on emptiness alone.
    """
    client, _, cloud = _hybrid({"resolutions": []}, {"resolutions": [{"x": 1}]}, monkeypatch)
    out = await client.generate_response([], response_model=NodeResolutions)
    assert out == {"resolutions": []} and not cloud.called


async def test_empty_extraction_without_credits_returns_the_empty_result(monkeypatch):
    """Degrade, never raise: the pipeline already flags 0-fact writes and queues them for review.

    Raising here would fail the whole write and lose the prose too.
    """
    client, _, cloud = _hybrid({"edges": []}, {"edges": [{"fact": "x"}]}, monkeypatch, credits=False)
    out = await client.generate_response([], response_model=ExtractedEdges)
    assert out == {"edges": []} and not cloud.called


async def test_the_cloud_retry_gets_pristine_messages(monkeypatch):
    """Graphiti's base generate_response MUTATES messages — it appends the response schema to the
    last message and language instructions to the first. Handing the cloud the SAME list after the
    local attempt would send those instructions twice. Pinned for both retry paths.
    """
    class _Msg:
        def __init__(self, content):
            self.content = content

        def model_copy(self, deep=False):
            return _Msg(self.content)

    class _Mutating:
        async def generate_response(self, messages, *a, **k):
            messages[0].content += " [schema appended by graphiti]"
            return {"edges": []}

    from synapse.core import llm_fallback
    monkeypatch.setattr(llm_fallback, "anthropic_available", lambda: True)

    seen: list[str] = []

    class _Cloud:
        async def generate_response(self, messages, *a, **k):
            seen.append(messages[0].content)
            return {"edges": [{"fact": "x"}]}

    client = HybridLLMClient(local=_Mutating(), cloud=_Cloud())
    await client.generate_response([_Msg("original prompt")], response_model=ExtractedEdges)
    assert seen == ["original prompt"], f"cloud received mutated input: {seen}"


def test_factory_selects_client_by_mode(monkeypatch):
    from synapse.core import extraction_clients as ec
    monkeypatch.setattr(ec, "build_cloud_client", lambda: object())   # avoid needing an API key

    monkeypatch.setattr(ec.settings, "extraction_mode", "hybrid")
    assert isinstance(ec.build_extraction_client(), HybridLLMClient)

    monkeypatch.setattr(ec.settings, "extraction_mode", "local")
    assert isinstance(ec.build_extraction_client(), OllamaStrictClient)

    monkeypatch.setattr(ec.settings, "extraction_mode", "cloud")
    assert build_extraction_client() is not None                      # the cloud stub
