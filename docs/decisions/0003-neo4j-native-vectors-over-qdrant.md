# 3. Neo4j's native vector index, not a separate Qdrant

**Status:** Accepted — 2026-05-31 (reverses an earlier "add Qdrant" decision)

## Context

The initial architecture included Qdrant as a dedicated vector store alongside
Neo4j — the conventional GraphRAG pattern of "graph in Neo4j, vectors in a
purpose-built ANN engine." Once retrieval was actually running on Graphiti's
Neo4j-backed hybrid search, that decision was worth re-opening: does a separate
Qdrant still earn its place?

Three Synapse-specific facts drove the re-evaluation:

1. **Scale.** Neo4j's native vector index (Lucene HNSW) is comfortable to roughly 1M
   vectors on a single instance; the standard "offload to Qdrant/Milvus" advice
   kicks in past ~10M. Synapse stores curated knowledge across ten projects — a
   realistic ceiling over years is thousands to low tens of thousands of facts,
   well under 2% of the single-instance comfort zone. Qdrant's entire value
   proposition (GPU ANN, billions of vectors, 10K QPS, sharding) is throughput we
   will never approach.
2. **Latency.** Retrieval happens at session start (the cached `brief`) and
   occasional `recall` — human-paced, not per-keystroke. Qdrant's few-millisecond
   edge over Neo4j is irrelevant when briefs are Redis-cached.
3. **Graphiti doesn't support an external vector store.** It writes embeddings into
   Neo4j *atomically with the graph*. External-store support is an open feature
   request, not shipped. Bolting on Qdrant means a parallel, unsupported embedding
   path and the classic dual-write sync problem — orphaned vectors, reconciliation
   jobs — which directly contradicts our single-transactional-store and
   don't-corrupt-the-graph rules.

## Decision

Drop Qdrant from the stack. Vectors live in Neo4j's native index via Graphiti
(BGE-M3, 1024-dim). Remove the Qdrant service and its client dependency; keep an
empty, documented seam so re-adding is low-friction.

## Consequences

**Positive.** One transactional store, no dual-write sync surface, no orphaned
vectors, less RAM and one fewer container. The graph+vector hybrid uplift is already
captured by Graphiti (cosine + BM25 + graph BFS over Neo4j) — Qdrant would have
added a duplicate copy of the same vectors, not new capability.

**Negative / accepted.** We forgo Qdrant's advanced filtered vector search and any
future billion-scale headroom. Our Neo4j is Community 5.26, so the newest in-index
filtering (GA in the 2026.x line) isn't available — but Graphiti doesn't require it
and our scale doesn't need it; upgrading Neo4j is a cleaner path than adding Qdrant.
The decision is explicitly reversible, gated on documented triggers: re-add Qdrant
only if the corpus approaches ~500K facts, we build a bespoke reranking layer
outside Graphiti, or Graphiti ships first-class external-vector-store support. This
is a case of deleting infrastructure that scale didn't justify — the right call is
often the one that removes a moving part.
