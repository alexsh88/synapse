# 6. Mode-configurable extraction: cloud / local / hybrid

**Status:** Accepted — 2026-06-06

## Context

Turning a raw text episode into typed entities and edges is the quality-critical
step of the write path — it decides what actually lands in the graph. Claude Sonnet
is the default extractor: high accuracy, reliable structured output. But it is the
only paid call in the stack, and the question arose whether bulk writes could move
to a local Ollama model for near-zero cost — sharpened when API credits ran out
mid-project and the system had to keep working without cloud access.

We ran an empirical A/B on 14 real Synapse episodes through Graphiti's full pipeline:

| Extractor | Ingestion success | Entities | Edges |
|---|---|---|---|
| Claude Sonnet | 14/14 (100%) | 102 | 118 |
| gemma3:12b (Graphiti default, loose JSON) | 10/14 (71%) | 58 (57%) | 50 (42%) |
| gemma3:12b + **strict `json_schema`** | 12/14 (86%) | 83 (81%) | 76 (64%) |

The decisive finding: **the risk is not accuracy, it's structured-output
reliability.** Local failures were malformed JSON that broke Graphiti's parser even
after retries — extracting *nothing* and **silently losing the facts**, clustered on
the richest episodes where extraction matters most. Sending the Pydantic schema as
`response_format={type: json_schema, strict: true}` cut failures (71%→86%) *and*
nearly doubled edge capture (42%→64%) — forcing schema adherence makes the model
extract more completely, not just more validly. But even strict decoding leaves a
non-trivial empty tail (roughly 14% of writes still lost facts), and a real edge-
density gap (a 12B model is genuinely worse at relationships) that config can't close.

## Decision

Make extraction **mode-configurable** via an `extraction_mode` setting:

- `cloud` (default) — Claude Sonnet; no quality regression, preserved as default.
- `local` — gemma3:12b via Ollama with strict `json_schema` decoding, $0.
- `hybrid` — local extraction with automatic Sonnet fallback on schema-validation
  failure or flagged high-value writes.

Empty extractions are detected and routed to a review queue rather than returning a
clean `STORED` with zero facts. A separate credit-aware fallback flips cloud→local
when Anthropic credits are exhausted, with the cooldown state shared across the ten
MCP processes via Redis so they don't each rediscover exhaustion independently.

## Consequences

**Positive.** Writes survive credit exhaustion — the system degrades to local
extraction instead of stopping. `hybrid` recovers near-Sonnet quality at near-zero
cost with a bounded failure risk. The strict-schema config is a reusable win for any
local extraction.

**Negative — honestly.** `local` accepts a real ~14% silent-empty-fact tail and a
sparser, less-connected graph; it is only acceptable if a few dropped facts are
tolerable. The Sonnet-vs-open quality delta for *our* task is inferred from general
benchmarks, not directly measured — no benchmark tests Sonnet on Graphiti's exact
job. Default stays `cloud` precisely because quality is the point; the local modes
are a cost/resilience lever, deliberately not the default.
