# 2. Local BGE-M3 embeddings via Ollama, not a hosted embedder

**Status:** Accepted — 2026-05-31

## Context

Graphiti needs an embedder to power semantic retrieval. The obvious default is a
hosted embedding API (OpenAI, Gemini, Voyage). Synapse's own operating principle is
self-hosted-first: own the data, keep bulk knowledge local, avoid managed
dependencies without a deliberate reason.

The cost axis turns out not to decide this. Embeddings are input-only and Synapse's
corpus is tiny — curated facts across ten projects, thousands to low tens of
thousands of nodes over years, not millions. Even the priciest hosted model at real
scale is roughly $18/month; at our volume any hosted option costs cents per *year*.
So "cheapest" is a wash. The deciding axes are: owning the data, re-embedding
lock-in, and quality.

We have the hardware — a 16 GB consumer GPU — so local embeddings are both free per
token and faster (roughly 15–50 ms on-GPU vs. 200–800 ms round-trip to a cloud
API).

## Decision

Run **BGE-M3 (1024-dim)** locally, served through Ollama via its OpenAI-compatible
endpoint, pointed at by Graphiti's embedder client. BGE-M3 is the model the
Graphiti/Zep research paper itself used for both embedding and reranking — the
lowest-risk pairing with the engine. It offers an 8K-token context (our knowledge
facts can be paragraph-length; `mxbai-embed-large`'s 512-token cap would truncate
them) and hybrid dense+sparse output that aligns with our multi-strategy retrieval.
No OpenAI dependency exists anywhere in the stack as a result; the only outbound
call is the extraction prompt to Claude.

## Consequences

**Positive.** Bulk knowledge embeddings never leave the machine. Zero per-token
cost, low latency, and a model proven against our exact engine. Model swaps stay a
config change because Ollama speaks the OpenAI wire format.

**Negative — the honest downsides.** The vector dimension is **locked at 1024 at
first ingestion**. Changing embedders later means re-embedding the entire corpus and
rebuilding every vector index — a genuine one-way door, chosen deliberately here.

Ollama also became a hard single point of failure for *all* writes: no embedding, no
store. And BGE-M3 on Ollama intermittently returned NaN embeddings, surfacing as a
cryptic `Ollama 500: "json: unsupported value: NaN"` that failed writes mid-session.
That is a known Ollama flash-attention numerical-instability bug, not a Synapse
defect; the fix was `OLLAMA_FLASH_ATTENTION=0` (a failing string went 15/15 → 0/20),
backed by a retrying embedder and, later, a queue-and-replay path so an embedder
outage degrades to a pending write instead of a hard failure. Running your own
embedder means owning your own infrastructure bugs — an accepted cost of self-hosting.
