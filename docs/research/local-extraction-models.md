# Research — Moving Graphiti extraction off Claude Sonnet to local Ollama models

**Date:** 2026-06-05 · **Question:** How much does extraction quality drop if Synapse moves Graphiti's
extraction LLM from Claude Sonnet 4.6 to a local open model (Ollama), and which free models are best?
**Method:** deep-research harness (5 angles, 22 sources, 25 claims verified 3-vote; 2 refuted 0-3).

## Bottom line
- **The real risk is NOT accuracy — it's structured-output reliability.** The dominant failure mode is
  malformed/mis-keyed JSON that fails Graphiti's Pydantic validation and **silently breaks ingestion**
  (facts never get stored). Graphiti's maintainers explicitly warn against smaller local models for
  exactly this reason. [high confidence]
- **Quality gap is bimodal:**
  - *General / personal-knowledge extraction:* near-zero at the top — Gemma3-27B **ties GPT-4o**
    (F1micro 0.96, composite 0.74) on LLMStructBench; token accuracy clusters ~0.89–0.96 from 0.6B→70B. [medium]
  - *Hard domain-specific KG / ontology:* large gap — older open 70B models retained only **~57–63%**
    of GPT-4 F1 (sepsis KG: GPT-4 76.8 vs Llama3-70B 48.4 / Qwen2-72B 43.8; SNOMED: GPT-4o 96.3 vs
    Llama3.3-70B 30.2). [high, but older models + harder task than Graphiti]
- **No benchmark directly measures Sonnet 4.6 vs current open models on Graphiti's exact task.** The
  drop for *Synapse specifically* is **inferred, not measured** → validate empirically on real episodes.
- **Honest estimate for Synapse's use (personal dev knowledge, short factual episodes):** with a strong
  14–32B model + proper structured-output config, expect a **small accuracy drop (~roughly 5–15%
  fewer/again-slightly-worse entities & edges)** but a **non-trivial ingestion-failure tail** (some % of
  writes silently lose facts) unless guarded with schema-validation + retry.

## Best free models (by VRAM tier, mid-2026)
| Tier | Model | Notes |
|---|---|---|
| **14–32B (sweet spot)** | **Gemma3-27B** | best *measured* general structured F1 (ties GPT-4o); ~18–22GB VRAM @ Q4 |
| 14–32B | **Qwen3-32B** | strong JSON/function reliability; ~20–24GB @ Q4 |
| 7–9B (budget) | Qwen2.5 / Qwen3-7–8B | runs anywhere, but **elevated JSON-failure risk** — constrained output mandatory |
| 70B+ (headroom) | Llama 3.3 70B / gpt-oss-120b | max quality; needs 48GB+ or multi-GPU |

Mid-size (12–27B) is the sweet spot; gains beyond ~14B are marginal, and **prompting/structured-output
config matters more than size**. ❌ **Refuted (0-3):** Graphiti does *not* officially bless
`deepseek-r1:7b` or `gpt-oss:120b` as reference local models — ignore blog claims that say so.

## Integration specifics for OUR stack (graphiti-core 0.29.1)
1. **Use `OpenAIGenericClient`**, not the default client — point it at Ollama's `/v1` chat-completions
   with `response_format` (same endpoint our embedder already uses). Ollama lacks the `v1 responses` API.
2. **Two-tier gotcha (version-specific):** Graphiti uses `model` **and** `small_model`. `small_model`
   defaults to `gpt-4.1-nano` (OpenAI) → **404 on Ollama**. Must set BOTH tiers (e.g. `SMALL_MODEL_NAME`
   / alias) or extraction breaks. [issue #1155, matches our 0.29.1]
3. **Mandatory guardrails:** temperature 0, constrained/schema output, JSON validation + **retry on
   Pydantic failure** (and ideally an escalation to Sonnet on repeated failure).

## Recommendation
**Hybrid is the safest high-value trade-off:** local 27–32B (Gemma3-27B / Qwen3-32B) for bulk/triage
writes, with **automatic fallback to Sonnet 4.6 only when local extraction fails schema validation or
on flagged high-value writes.** This recovers near-Sonnet quality at near-zero cost and bounds the
ingestion-failure risk. A full local swap (Gemma3-27B + strict structured output + retry) is viable for
personal knowledge if a few silently-dropped facts are acceptable.

**Before committing:** run an A/B on a held-out set of ~30–50 real Synapse episodes — re-extract with the
local model, diff entities/edges vs the Sonnet baseline, and measure the **ingestion-failure rate** (the
metric that actually matters). That converts the inferred drop into a measured one for *our* data.

## EMPIRICAL A/B on real Synapse episodes (2026-06-06, `scripts/ab_extract.py`)
14 real episodes (2 per project seed file) run through Graphiti's full pipeline under each LLM,
into throwaway groups (cleaned up after). Hardware: RTX 4070 Ti SUPER 16GB. Embedder bge-m3 shared.

| Model | Ingestion success | Nodes | Edges | Latency |
|---|---|---|---|---|
| **Claude Sonnet 4.6** | 14/14 (100%) | 102 | 118 | 43 s/ep |
| Gemma3-12B (loose, Graphiti default) | 10/14 (71%) | 58 (57%) | 50 (42%) | 126 s/ep |
| **Gemma3-12B + strict json_schema** | **12/14 (86%)** | **83 (81%)** | **76 (64%)** | 127 s/ep |
| Qwen3-14B (Ollama) | ~100% (0 fails in 9 eps) | low (~3-6/ep) | low | 100-520 s/ep (**unusable, reasoning mode**) |

**Decisive finding — strict json_schema (`OllamaStrictClient` in `scripts/ab_extract.py`):** sending the
Pydantic schema as `response_format={type:json_schema, json_schema:{...,strict:true}}` (the Graphiti
generic client omits `strict`) **cut failures (71%→86%) AND nearly doubled edge capture (42%→64%, entities
57%→81%)** — forcing schema adherence makes the model extract more *completely*, not just more *validly*.
This config should be the default for any local extraction. 2/14 dense episodes still failed even with
strict (Ollama grammar enforcement isn't perfect on big schemas) → a Sonnet fallback is still needed.
Qwen3-14B: reliable JSON but reasoning-mode latency (up to 9 min/ep) makes it impractical without
disabling thinking.

**Findings (confirm the research):**
- The killer is **structured-output reliability, not accuracy.** Gemma's 4 failures were all malformed
  JSON (unterminated strings, missing delimiters) that broke Graphiti's parser **even after retries** →
  zero extracted, knowledge **silently lost**. Failures clustered on the *richest* episodes (acme-flow,
  acme-docs, acme-etl) — exactly where extraction matters most.
- Even on the 10 successes, Gemma captured ~57% of entities and only **~42% of relationships** vs Sonnet
  (edges are the harder task) → a much sparser, less-connected graph.
- ~3× slower (inflated by retry storms on the JSON failures).

**Verdict: full-local Gemma3-12B as-is is NOT acceptable** (~29% silent write loss + half the connectivity).
Two levers: (1) **schema-constrained decoding** (Ollama `format=<json_schema>`) would *guarantee* valid
JSON and likely kill most of the 29% failure rate — a config fix, not a model limit; (2) the ~50% edge
gap is a genuine 12B-capability gap that config can't close. ⇒ **Hybrid (local + Sonnet fallback on
JSON-validation failure) is the right architecture**, OR invest in strict structured-output enforcement
and accept lower edge density for $0.

## IMPLEMENTED (2026-06-06) — `extraction_mode` flag + hybrid client
`synapse/core/extraction_clients.py`: `OllamaStrictClient` (strict json_schema local) + `HybridLLMClient`
(local → Sonnet fallback) + `build_extraction_client()`. Selected by `settings.extraction_mode`
(`cloud` default / `hybrid` / `local`) and `local_extraction_model` (default gemma3:12b); wired into
`build_graphiti()`. Default stays **cloud** (no quality regression). **Enable hybrid:** set
`EXTRACTION_MODE=hybrid` in `.env` (host) or the `api` service env (compose), then restart the API.
Verified live: a hybrid `remember` extracted entities+edges via local gemma3:12b, Sonnet untouched.
Tests: `tests/test_extraction_clients.py`; live smoke: `scripts/hybrid_smoke.py`.

## Caveats
- No comparator is Sonnet 4.6 (all use GPT-4 family) → the Sonnet-vs-open delta is inferred.
- LLMStructBench (the most open-favorable evidence) is generic JSON, not Graphiti KG, and split 2-1.
- Document-level full-JSON validity (≈ ingestion success) is far lower & more prompt-sensitive than token
  F1 even for top models — ingestion-failure risk is real regardless of model.
- CLAUDE.md locks Sonnet 4.6 as the extraction model ("only outbound API call"; "no managed deps without
  discussion") — moving to local is a **deliberate relitigation of a locked decision**, done on purpose here.

## Sources (primary)
Zep/Graphiti: help.getzep.com/graphiti/configuration/llm-configuration · github.com/getzep/graphiti
(+ mcp_server/README, issues #796/#1116/#1155/#1204) · blog.getzep.com/llm-rag-knowledge-graphs-faster-and-more-dynamic
Benchmarks: arxiv.org/abs/2602.14743 (LLMStructBench) · arxiv.org/abs/2510.11297 · PMC11986385 (sepsis KG) ·
PMC12061982 (SNOMED) · platform.claude.com/cookbook/capabilities-knowledge-graph-guide
