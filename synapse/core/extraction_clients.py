"""Extraction LLM clients — cloud (Sonnet), strict-local (Ollama), and hybrid (cost routing).

The extraction LLM is Synapse's only paid dependency. `settings.extraction_mode` selects:

- **cloud**  — Claude Sonnet 4.6 (default; the locked-decision quality baseline).
- **local**  — `local_extraction_model` (default gemma3:12b) via Ollama with STRICT json_schema
  decoding. Measured on real episodes (docs/research/local-extraction-models.md): ~81%/64% of
  Sonnet's entities/edges and ~14% of dense writes silently fail — so prefer hybrid.
- **hybrid** — try strict-local first; fall back to Sonnet on any extraction failure (bad JSON /
  schema / timeout) AND on an empty node/edge extraction, which is what a strict-decoded local
  failure actually looks like. Captures the ~$0 majority without silently losing the local tail.

Why strict matters: Graphiti's generic Ollama client sends a json_schema WITHOUT `strict`, so Ollama
treats it as a hint and the model can emit broken JSON. Forcing `strict:true` switches Ollama to hard
grammar-constrained decoding — in our A/B that cut failures 71%→86% AND nearly doubled edge capture.
"""

from __future__ import annotations

import json
import logging
from typing import Any, cast

from graphiti_core.llm_client.client import LLMClient
from graphiti_core.llm_client.config import LLMConfig, ModelSize
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient

from synapse.config import settings

logger = logging.getLogger("synapse.extraction")


class OllamaStrictClient(OpenAIGenericClient):
    """Local extraction via Ollama with STRICT schema-constrained decoding (guaranteed-valid JSON)."""

    async def _generate_response(self, messages, response_model=None,
                                 max_tokens: int = 16384, model_size: ModelSize = ModelSize.medium):
        msgs: list[Any] = []
        for m in messages:
            m.content = self._clean_input(m.content)
            if m.role in ("user", "system"):
                msgs.append({"role": m.role, "content": m.content})
        rf: dict[str, Any]
        if response_model is not None:
            rf = {"type": "json_schema", "json_schema": {
                "name": getattr(response_model, "__name__", "resp"),
                "schema": response_model.model_json_schema(), "strict": True}}
        else:
            rf = {"type": "json_object"}
        # LLMConfig.model is Optional, but a request without one is a 400 from Ollama with a
        # message that names neither this client nor the config that produced it.
        if not self.model:
            raise RuntimeError("OllamaStrictClient needs a model name in its LLMConfig")
        # cast: the OpenAI SDK types these as TypedDicts, and both are assembled dynamically
        # here (the schema comes from a runtime pydantic model). The shapes are correct.
        resp = await self.client.chat.completions.create(
            model=self.model, messages=cast(Any, msgs), temperature=self.temperature,
            max_tokens=self.max_tokens, response_format=cast(Any, rf))
        return json.loads(resp.choices[0].message.content or "")


# Response models for which an EMPTY result is a suspected local failure rather than a real answer.
# Deliberately narrow: graphiti uses these two ONLY for node and edge extraction, while dedupe and
# invalidation prompts have their own models and legitimately answer "nothing" all the time —
# escalating those would buy a Sonnet call to confirm a correct empty answer.
_EMPTY_IS_SUSPECT = {"ExtractedEntities", "ExtractedEdges"}


def _is_empty_extraction(result) -> bool:
    """True when every list field came back empty — the shape of "the model found nothing"."""
    if not isinstance(result, dict):
        return False
    lists = [value for value in result.values() if isinstance(value, list)]
    return bool(lists) and all(not value for value in lists)


class HybridLLMClient(LLMClient):
    """Try local extraction; fall back to cloud (Sonnet) when local fails OR returns nothing.

    The "or returns nothing" half was missing until 2026-07-27, and it made this class's own
    promise false. STRICT json_schema decoding guarantees the local model emits *valid* JSON, not
    *useful* JSON: on dense input gemma returns a well-formed but EMPTY extraction, no exception is
    raised, and the old code returned that empty result as though it were an answer. The write is
    then stored with 0 facts and `degraded=True` — prose in the graph that no recall can reach.

    Measured on one 2.6KB five-paragraph research item, same content, same minute, credits
    available: hybrid 0 facts, cloud 15. A 1.5KB single-topic lesson extracted 13 under hybrid, so
    the failure is content-dependent, which is exactly why it survived the A/B (86% ingest) — the
    tail was silent rather than loud.
    """

    def __init__(self, local: LLMClient, cloud: LLMClient) -> None:
        super().__init__(config=LLMConfig(), cache=False)
        self.local = local
        self.cloud = cloud

    async def generate_response(self, messages, response_model=None, *args, **kwargs):
        from synapse.core import cost
        from synapse.core.llm_fallback import anthropic_available

        # Graphiti's base generate_response MUTATES messages in place — it appends the serialized
        # response schema to the last message and language instructions to the first. So the cloud
        # retry must start from a pristine copy, or Sonnet receives those instructions twice.
        pristine = [m.model_copy(deep=True) for m in messages]
        step = getattr(response_model, "__name__", "") or "extraction"

        # Attribution only, not tokens: Graphiti owns these clients and does not surface usage back
        # to the caller, so the honest record is which provider served which extraction step. That
        # is still the number the extraction_mode decision turns on — it makes the local-vs-cloud
        # mix, and therefore the escalation rate, visible for the first time.
        async def served_by(provider: str, model: str) -> None:
            await cost.track_call(operation=f"extract:{step}", model=model, provider=provider)

        try:
            result = await self.local.generate_response(messages, response_model, *args, **kwargs)
        except Exception as exc:  # noqa: BLE001 — a local failure is expected ~14% of the time
            if not anthropic_available():
                raise  # cloud is out of credits too — don't make a guaranteed-failing call
            logger.info("local extraction failed (%s); falling back to Sonnet", str(exc)[:100])
            await served_by("anthropic", settings.extraction_model)
            return await self.cloud.generate_response(pristine, response_model, *args, **kwargs)

        name = getattr(response_model, "__name__", "")
        if name in _EMPTY_IS_SUSPECT and _is_empty_extraction(result):
            if not anthropic_available():
                # Nothing better on offer. Return the empty result rather than raising: the write
                # pipeline already flags 0-fact writes as degraded and queues them for review.
                logger.warning("local extraction returned nothing for %s and cloud is unavailable; "
                               "the write will be flagged degraded", name)
                await served_by("ollama", settings.local_extraction_model)
                return result
            logger.info("local extraction returned an EMPTY %s; escalating to Sonnet", name)
            await served_by("anthropic", settings.extraction_model)
            return await self.cloud.generate_response(pristine, response_model, *args, **kwargs)
        await served_by("ollama", settings.local_extraction_model)
        return result

    async def _generate_response(self, messages, response_model=None,
                                 max_tokens=None, model_size: ModelSize = ModelSize.medium):
        # Not used (generate_response is overridden), but required by the ABC.
        return await self.local._generate_response(messages, response_model, max_tokens, model_size)


def build_local_client() -> OllamaStrictClient:
    return OllamaStrictClient(config=LLMConfig(
        api_key="ollama", model=settings.local_extraction_model,
        small_model=settings.local_extraction_model,
        base_url=settings.ollama_base_url, temperature=0))


def build_cloud_client() -> LLMClient:
    from synapse.core.knowledge_engine import build_llm_client  # lazy: avoids a circular import
    return build_llm_client()


def build_extraction_client() -> LLMClient:
    """The extraction LLM client for the configured mode."""
    mode = settings.extraction_mode
    if mode == "local":
        logger.info("extraction: LOCAL (%s, strict)", settings.local_extraction_model)
        return build_local_client()
    if mode == "hybrid":
        logger.info("extraction: HYBRID (%s strict → Sonnet fallback)", settings.local_extraction_model)
        return HybridLLMClient(local=build_local_client(), cloud=build_cloud_client())
    return build_cloud_client()
