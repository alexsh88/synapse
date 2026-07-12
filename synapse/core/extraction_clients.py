"""Extraction LLM clients — cloud (Sonnet), strict-local (Ollama), and hybrid (cost routing).

The extraction LLM is Synapse's only paid dependency. `settings.extraction_mode` selects:

- **cloud**  — Claude Sonnet 4.6 (default; the locked-decision quality baseline).
- **local**  — `local_extraction_model` (default gemma3:12b) via Ollama with STRICT json_schema
  decoding. Measured on real episodes (docs/research/local-extraction-models.md): ~81%/64% of
  Sonnet's entities/edges and ~14% of dense writes silently fail — so prefer hybrid.
- **hybrid** — try strict-local first; on ANY extraction failure (bad JSON / schema / timeout) fall
  back to Sonnet. Captures the ~$0 majority while never silently losing a fact to the local tail.

Why strict matters: Graphiti's generic Ollama client sends a json_schema WITHOUT `strict`, so Ollama
treats it as a hint and the model can emit broken JSON. Forcing `strict:true` switches Ollama to hard
grammar-constrained decoding — in our A/B that cut failures 71%→86% AND nearly doubled edge capture.
"""

from __future__ import annotations

import json
import logging

from graphiti_core.llm_client.client import LLMClient
from graphiti_core.llm_client.config import LLMConfig, ModelSize
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient

from synapse.config import settings

logger = logging.getLogger("synapse.extraction")


class OllamaStrictClient(OpenAIGenericClient):
    """Local extraction via Ollama with STRICT schema-constrained decoding (guaranteed-valid JSON)."""

    async def _generate_response(self, messages, response_model=None,
                                 max_tokens: int = 16384, model_size: ModelSize = ModelSize.medium):
        msgs = []
        for m in messages:
            m.content = self._clean_input(m.content)
            if m.role in ("user", "system"):
                msgs.append({"role": m.role, "content": m.content})
        if response_model is not None:
            rf = {"type": "json_schema", "json_schema": {
                "name": getattr(response_model, "__name__", "resp"),
                "schema": response_model.model_json_schema(), "strict": True}}
        else:
            rf = {"type": "json_object"}
        resp = await self.client.chat.completions.create(
            model=self.model, messages=msgs, temperature=self.temperature,
            max_tokens=self.max_tokens, response_format=rf)
        return json.loads(resp.choices[0].message.content or "")


class HybridLLMClient(LLMClient):
    """Try local extraction; on any failure, fall back to the cloud (Sonnet) client."""

    def __init__(self, local: LLMClient, cloud: LLMClient) -> None:
        super().__init__(config=LLMConfig(), cache=False)
        self.local = local
        self.cloud = cloud

    async def generate_response(self, messages, response_model=None, *args, **kwargs):
        from synapse.core.llm_fallback import anthropic_available

        try:
            return await self.local.generate_response(messages, response_model, *args, **kwargs)
        except Exception as exc:  # noqa: BLE001 — a local failure is expected ~14% of the time
            if not anthropic_available():
                raise  # cloud is out of credits too — don't make a guaranteed-failing call
            logger.info("local extraction failed (%s); falling back to Sonnet", str(exc)[:100])
            return await self.cloud.generate_response(messages, response_model, *args, **kwargs)

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
