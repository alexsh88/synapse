# Research — Embedder Provider Decision (§6B)

**Date:** 2026-05-31
**Question:** Best + cheapest embedder for Synapse, respecting R9 (self-hosted, own the data).
**Recommendation:** ✅ **Local BGE-M3, GPU-accelerated, served via Ollama** (1024-dim). Free per token, private, and it's the model Graphiti's own research paper used. Pair with Claude Sonnet 4.6 for extraction and Graphiti's built-in reranker for the MVP.

---

## The key realization: cost is a non-factor here

Embeddings are **input-only** and Synapse's volume is tiny (knowledge facts across ~10 projects — thousands, not millions, of nodes). Even the *most expensive* hosted model at real scale (100M tok/mo) is ~$18/mo; at Synapse's volume, **any hosted option costs cents per year.** So "cheapest" doesn't decide this — the deciding axes are **R9 (own the data)**, **re-embedding lock-in**, and **quality**.

And we have the hardware: **NVIDIA RTX 4070 Ti SUPER, 16 GB VRAM.** Local embeddings are therefore both **free per token** *and* fast (15–50 ms on-GPU vs 200–800 ms for cloud). Local wins on cost (tie at ~free) *and* on R9 *and* on privacy. That's the whole argument.

## Three components, not one (this is what the cost question really touches)

Graphiti needs **LLM (extraction)** + **embedder** + **cross-encoder/reranker**. The default reranker is an *LLM-based* OpenAI reranker — which is why naive setups "still need an OPENAI_API_KEY even with Ollama." Verified against the installed `graphiti-core==0.29.1`:

| Component | Modules shipped in 0.29.1 | Synapse choice |
|-----------|---------------------------|----------------|
| LLM (extraction) | `anthropic_client` ✓, openai, openai_generic, gemini, groq, gliner2 | **Claude Sonnet 4.6** (§6A — quality-critical, low volume, structured output) |
| Embedder | openai, azure_openai, gemini, voyage | **BGE-M3 via OpenAI-compatible client → Ollama** (see runtime note) |
| Cross-encoder | `bge_reranker_client` ✓ (local), gemini_reranker, openai_reranker | **Built-in RRF / node-distance for MVP; add local `BGERerankerClient` later** |

> Result: with Anthropic LLM + local BGE embedder + built-in/BGE reranker, **no OpenAI key is required anywhere.** The only thing leaving the machine is the extraction prompt to Claude (our own knowledge text, low volume) — an already-accepted managed dependency. Embeddings (high-volume, re-embedding-sensitive, bulk-sensitive) stay 100% local. Maximally R9-aligned.

## Why BGE-M3 specifically

- **Graphiti-proven:** the official Zep/Graphiti research paper used **BGE-M3 for both embedding and reranking.** Lowest-risk pairing with the engine.
- **1024-dim, 8K-token context** — knowledge facts can be paragraph-length; BGE-M3 handles them (vs `mxbai-embed-large`'s 512-token cap, which truncates and degrades sharply past ~1K tokens).
- **Hybrid dense+sparse** in one model — aligns with our multi-strategy retrieval (plan Part 5: semantic + keyword + graph).
- **Multilingual** — future-proof, near-free.
- **Fits the GPU easily** (~1.2 GB; 16 GB VRAM has room for a local reranker and even a local LLM later).

### Embedding quality reference (MTEB, 2026)
| Model | MTEB | Context | Size | Notes |
|-------|------|---------|------|-------|
| `nomic-embed-text` | 62.4 | 8192 | ~300MB | ≈ OpenAI 3-small; great for short/specific queries; lighter fallback |
| `mxbai-embed-large` | 64.7 | **512** ⚠️ | ~700MB | ≈ 3-large but context too short for our facts |
| **BGE-M3** | ~64 (1024-dim) | 8192 | ~1.2GB | **hybrid, multilingual, Graphiti's choice — pick this** |
| `qwen3-embedding:8b` | 70.6 | long | 4–6GB(Q4) | best quality; overkill now, but our GPU *could* run it |

Hosted, for comparison (all break R9, all create provider lock-in): Google text-embedding-005 $0.006/M (but 2K context cap), OpenAI 3-small $0.02/M, Voyage-4-lite $0.02/M, Gemini-embedding-001 $0.15/M (best hosted MTEB 68.3).

## Runtime: how to actually run BGE-M3 locally

`graphiti-core` 0.29.1's **embedder package has no in-process sentence-transformers class** (only openai/azure/gemini/voyage). Two viable paths:

1. **Ollama (recommended)** — `ollama pull bge-m3`, then point Graphiti's `OpenAIEmbedder` at `http://localhost:11434/v1` (dummy api_key). Pros: no torch in our venv, Ollama auto-uses the GPU, OpenAI-compatible so model swaps are config, and Ollama can later host a local LLM for fully-air-gapped extraction. Con: one more local service (Ollama is **not currently installed** — would need install).
2. **Custom in-process embedder** — subclass `EmbedderClient` wrapping `FlagEmbedding`/sentence-transformers BGE-M3. Pros: no external service. Cons: pulls ~2GB torch into our venv, more code to maintain.

→ **Go with Ollama.** Cleaner separation, GPU-managed, and it doubles as the future local-LLM host.

## Lock-in warning

Whatever we choose, the **vector dimension is fixed** at first ingestion (Qdrant collection + Neo4j vector index). Switching embedders later = re-embed everything. BGE-M3's **1024-dim** is the committed value. Decide once, here.

## Recommended Synapse embedding stack

```
Extraction LLM : Claude Sonnet 4.6      (Anthropic API — only outbound call)
Embedder       : BGE-M3 (1024-dim)      (local, Ollama, GPU — $0/token)
Reranker       : built-in RRF / node-distance  (MVP)
                 → local BGERerankerClient later (quality, still $0)
Vector dim     : 1024  (LOCKED at first ingestion)
```

**Cost:** embeddings + reranking = **$0/token** (own GPU). Only Sonnet extraction incurs API cost, at low volume.
**R9:** fully honored — bulk knowledge embeddings never leave the machine.

---

## Sources
- [Graphiti LLM/embedder configuration (Zep docs)](https://help.getzep.com/graphiti/configuration/llm-configuration) · [Graphiti embedding & reranking (DeepWiki)](https://deepwiki.com/getzep/graphiti/6.2-embedding-and-reranking-services) · [Zep temporal KG paper (uses BGE-M3)](https://arxiv.org/pdf/2501.13956) · [getzep/graphiti](https://github.com/getzep/graphiti)
- [Best local embedding models 2026 / Ollama](https://www.morphllm.com/ollama-embedding-models) · [Open-source embeddings guide (BentoML)](https://www.bentoml.com/blog/a-guide-to-open-source-embedding-models) · [Embedding benchmark 2026 (Cheney Zhang)](https://zc277584121.github.io/rag/2026/03/20/embedding-models-benchmark-2026.html) · [Milvus: choose embedding model 2026](https://milvus.io/blog/choose-embedding-model-rag-2026.md)
- [Embedding model pricing 2026 (PricePerToken)](https://pricepertoken.com/embedding) · [Text embedding models 2026: Google vs OpenAI vs Voyage (TokenMix)](https://tokenmix.ai/blog/text-embedding-models-comparison) · [Ollama OpenAI compatibility](https://docs.ollama.com/api/openai-compatibility)
