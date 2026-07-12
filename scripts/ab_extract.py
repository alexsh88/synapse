"""A/B: Graphiti extraction — local Ollama models vs Claude Sonnet (cost research, 2026-06).

Adds a fixed sample of REAL Synapse episodes (from scripts/seeds/*.json) to throwaway group_ids
under each LLM, records ingestion success + #entities + #edges + latency per episode, prints a
comparison vs the Sonnet baseline, then deletes the throwaway groups. Measures the metric that
actually matters: how many writes a local model successfully ingests (vs silently failing) and how
its extraction volume compares to Sonnet's.

    python -m scripts.ab_extract                    # sonnet + gemma3:12b + qwen3:14b
    python -m scripts.ab_extract gemma3:12b         # one model (still needs sonnet for the baseline row)
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from graphiti_core import Graphiti
from graphiti_core.llm_client.config import LLMConfig, ModelSize
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
from graphiti_core.nodes import EpisodeType


class OllamaStrictClient(OpenAIGenericClient):
    """Generic client that forces Ollama's NATIVE schema-constrained decoding (Ollama `format` field
    via extra_body) instead of the loose `json_object` Graphiti sends — so the model is GRAMMAR-forced
    to emit syntactically valid JSON matching the Pydantic schema. Tests whether the malformed-JSON
    ingestion failures are a config problem (fixable) vs a model-capability problem (not)."""

    async def _generate_response(self, messages, response_model=None,
                                 max_tokens: int = 16384, model_size: ModelSize = ModelSize.medium):
        msgs = []
        for m in messages:
            m.content = self._clean_input(m.content)
            if m.role in ("user", "system"):
                msgs.append({"role": m.role, "content": m.content})
        # Proper OpenAI-compatible structured output WITH strict — Ollama maps a strict json_schema
        # response_format to hard grammar-constrained decoding (the base client omits strict).
        if response_model is not None:
            rf = {"type": "json_schema", "json_schema": {
                "name": getattr(response_model, "__name__", "resp"),
                "schema": response_model.model_json_schema(), "strict": True}}
        else:
            rf = {"type": "json_object"}
        resp = await self.client.chat.completions.create(
            model=self.model, messages=msgs,
            temperature=self.temperature, max_tokens=self.max_tokens, response_format=rf,
        )
        return json.loads(resp.choices[0].message.content or "")

from synapse.config import settings
from synapse.core.knowledge_engine import PassthroughCrossEncoder, build_embedder, build_llm_client
from synapse.core.schema import EDGE_TYPE_MAP, EDGE_TYPES, ENTITY_TYPES

REF = datetime(2026, 1, 1, tzinfo=timezone.utc)
SEEDS_DIR = Path(__file__).resolve().parent / "seeds"


def sample_episodes(n_per_file: int = 2) -> list[tuple[str, str]]:
    """A reproducible cross-project sample: first N facts from each seed file."""
    eps: list[tuple[str, str]] = []
    for f in sorted(SEEDS_DIR.glob("*.json")):
        facts = json.loads(f.read_text(encoding="utf-8"))
        for fact in facts[:n_per_file]:
            eps.append((f.stem, fact["content"] if isinstance(fact, dict) else str(fact)))
    return eps


def build(label: str) -> Graphiti:
    if label == "sonnet":
        llm = build_llm_client()                      # AnthropicClient (claude-sonnet-4-6)
        max_co = 4
    else:
        # Local via Ollama. A "-strict" suffix forces native schema-constrained decoding.
        strict = label.endswith("-strict")
        model = label[:-7] if strict else label
        cfg = LLMConfig(api_key="ollama", model=model, small_model=model,
                        base_url=settings.ollama_base_url, temperature=0)
        llm = (OllamaStrictClient if strict else OpenAIGenericClient)(config=cfg)
        max_co = 1                                     # one 12-14B model on a 16GB GPU — no concurrency
    return Graphiti(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password,
                    llm_client=llm, embedder=build_embedder(),
                    cross_encoder=PassthroughCrossEncoder(), max_coroutines=max_co)


async def run_model(label: str, episodes: list[tuple[str, str]]) -> list[dict]:
    g = build(label)
    group = f"abtest_{label.replace(':', '_')}"
    rows: list[dict] = []
    try:
        for i, (proj, text) in enumerate(episodes):
            t0 = time.monotonic()
            try:
                res = await g.add_episode(
                    name=f"ab-{i}", episode_body=text, source=EpisodeType.text,
                    source_description="abtest", reference_time=REF, group_id=group,
                    entity_types=ENTITY_TYPES, edge_types=EDGE_TYPES, edge_type_map=EDGE_TYPE_MAP)
                nodes = len(getattr(res, "nodes", []) or [])
                edges = len(getattr(res, "edges", []) or [])
                row = {"proj": proj, "ok": nodes > 0, "nodes": nodes, "edges": edges,
                       "secs": round(time.monotonic() - t0, 1), "err": None}
            except Exception as exc:  # noqa: BLE001 — a failed episode IS the measurement
                row = {"proj": proj, "ok": False, "nodes": 0, "edges": 0,
                       "secs": round(time.monotonic() - t0, 1), "err": str(exc).replace("\n", " ")[:140]}
            rows.append(row)
            print(f"  [{label}] ep{i:02} {proj:22} ok={str(row['ok']):5} n={row['nodes']:2} e={row['edges']:2} "
                  f"{row['secs']:5}s" + (f"  ERR {row['err']}" if row["err"] else ""), flush=True)
    finally:
        await g.close()
    return rows


async def cleanup() -> None:
    g = build("sonnet")
    try:
        await g.driver.execute_query("MATCH (n) WHERE n.group_id STARTS WITH 'abtest_' DETACH DELETE n")
    finally:
        await g.close()


async def main(models: list[str]) -> int:
    eps = sample_episodes()
    print(f"A/B extraction on {len(eps)} real episodes; models={models}\n", flush=True)
    results: dict[str, list[dict]] = {}
    for label in models:
        print(f"== {label} ==", flush=True)
        results[label] = await run_model(label, eps)

    print("\n=== SUMMARY ===")
    base = results.get("sonnet")
    bn = max(1, sum(r["nodes"] for r in base)) if base else 1
    be = max(1, sum(r["edges"] for r in base)) if base else 1
    for label, rows in results.items():
        ok = sum(r["ok"] for r in rows)
        tn, te = sum(r["nodes"] for r in rows), sum(r["edges"] for r in rows)
        avg = round(sum(r["secs"] for r in rows) / len(rows), 1)
        line = f"{label:14} ingest {ok}/{len(rows)} | nodes {tn:3} edges {te:3} | avg {avg:5}s/ep"
        if base and label != "sonnet":
            line += f" | vs sonnet: nodes {round(100*tn/bn)}% edges {round(100*te/be)}%"
        print(line)

    await cleanup()
    print("\ncleaned up abtest_* groups")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main(sys.argv[1:] or ["sonnet", "gemma3:12b", "qwen3:14b"])))
