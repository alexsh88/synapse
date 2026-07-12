# Research — Qdrant vs Neo4j-native vectors for Synapse

**Date:** 2026-05-31
**Question:** Now that retrieval runs on Graphiti's Neo4j-backed hybrid search, does a separate Qdrant earn its place? (Re-opening the locked "Vector store: Qdrant" decision.)
**Recommendation:** ✅ **Drop Qdrant from the MVP stack.** Zero functional loss at Synapse's scale, removes a real reliability cost (dual-write sync), and aligns with how Graphiti actually works. Fully reversible via a documented trigger.

---

## The decision hinges on three Synapse-specific facts

### 1. Scale — Synapse is ~3 orders of magnitude below where Qdrant matters
- Neo4j's native vector index (Lucene HNSW) is comfortable to **~1M vectors per index** on a single instance; the "offload to Qdrant/Milvus" advice kicks in **>10M vectors**.
- Synapse stores *curated* knowledge — decisions, conventions, lessons, research — across ~10 projects. Realistic ceiling over **years** is thousands to low tens of thousands of facts. That's **<1–2%** of Neo4j's comfortable single-instance ceiling.
- Qdrant's entire value prop (GPU ANN, billions of vectors, 10K QPS, aggressive quantization, horizontal sharding) is **scale/throughput we will never approach.**

### 2. Latency — Synapse is not latency-sensitive
- Retrieval happens at **session start** (`brief`, cached in Redis) and occasional `recall` — not per-keystroke. Qdrant's ~4ms-vs-Neo4j edge is irrelevant when briefs are cached and queries are human-paced.

### 3. Graphiti doesn't support an external vector store (and the sync cost is real)
- External vector-store support in Graphiti is an **open feature *request* (Feb 2026, issue #1263), not shipped.** Graphiti writes embeddings into Neo4j *atomically with the graph*. Bolting on Qdrant means a **parallel, unsupported embedding path** — swimming against the engine.
- The dominant industry warning about separate vector DBs is the **sync problem**: dual-write, orphaned vectors, reconciliation jobs. That directly contradicts **R3** (single transactional store, concurrent-write safety) and **R8** (don't corrupt the graph). Adding Qdrant would *introduce* the exact failure surface our rules exist to avoid — for no benefit at our scale.

## Would Qdrant improve retrieval *quality*? No — we already have the hybrid.
The cited accuracy uplift from Qdrant (e.g. Lettria +20–25%) is **vector-RAG vs graph+vector hybrid**. Synapse *already runs the graph+vector hybrid* via Graphiti (cosine + BM25 + graph BFS over Neo4j). That uplift is already captured; a parallel Qdrant adds a duplicate copy of the same vectors, not new capability.

## The "use both" pattern is for a different problem
Qdrant-for-vectors + Neo4j-for-graph is the right call for **large-scale GraphRAG** (10M+ vectors, high QPS, multi-tenant filtering). Synapse is a small, curated, single-user-ish knowledge brain. Different regime.

## Caveat (honest)
- Our Neo4j is **Community 5.26.24**, not the 2026.x line — so the newest *in-index filtering* (GA in Neo4j 2026.02) isn't available to us. Graphiti doesn't require it, and our scale doesn't need it. If we ever want advanced filtered vector search, **upgrading Neo4j** is a cleaner path than adding Qdrant.
- Numbers like the ~1M ceiling come from third-party benchmarks; methodology varies. But we're so far below any threshold that the margin is overwhelming.

## Recommendation: drop Qdrant now, keep the door open

**Do:**
1. Remove the `qdrant` service from `docker-compose.yml` (frees ports 6333/6334 + RAM).
2. Update the locked decision: *"Vectors live in Neo4j's native index via Graphiti (BGE-M3, 1024-dim). Qdrant removed — redundant at Synapse's scale."*
3. Keep `synapse/db/vector_client.py` as an empty, documented seam.

**Revisit (add Qdrant back) only if any of these fire:**
- Knowledge approaches **~500K+ facts** (still half the single-instance ceiling — early-warning line), or
- We build a **bespoke semantic/reranking layer outside Graphiti** (e.g. a UI-driven cross-project search Graphiti doesn't expose), or
- **Graphiti ships first-class external-vector-store support** and we want GPU ANN.

Re-adding is low-friction: store vector IDs as node properties in Neo4j and join after retrieval (the standard shared-ID pattern). Nothing we do now blocks it.

---

## Sources
- [Qdrant vs Neo4j vector search (Zilliz)](https://zilliz.com/blog/qdrant-vs-neo4j-a-comprehensive-vector-database-comparison) · [Neo4j vector index limits / ~1M sweet spot, >10M offload (Markaicode)](https://markaicode.com/architecture/neo4j-hybrid-retrieval-architecture/) · [Neo4j vector index memory config (Neo4j docs)](https://neo4j.com/docs/operations-manual/current/performance/vector-index-memory-configuration/)
- [Vector search with filters in Neo4j 2026.01 (Neo4j blog)](https://neo4j.com/blog/genai/vector-search-with-filters-in-neo4j-v2026-01-preview/) · [Graphiti external vector store feature request #1263](https://github.com/getzep/graphiti/issues/1263)
- [When a separate vector DB is worth it / the sync problem (Encore)](https://encore.dev/articles/pgvector-vs-qdrant) · [pgvector vs Qdrant decision framework (open-techstack)](https://open-techstack.com/blog/pgvector-vs-qdrant-2026/) · [Integrate Qdrant + Neo4j RAG (Neo4j dev blog)](https://neo4j.com/blog/developer/qdrant-to-enhance-rag-pipeline/) · [Lettria GraphRAG case study (Qdrant)](https://qdrant.tech/blog/case-study-lettria-v2/)
