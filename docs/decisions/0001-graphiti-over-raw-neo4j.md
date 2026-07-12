# 1. Use Graphiti as the temporal graph engine, not raw Neo4j

**Status:** Accepted — 2026-05-30

## Context

Synapse is a shared memory for AI coding agents: a knowledge graph that ten
connected projects read from and write to. The core requirement is *temporal* —
knowledge supersedes rather than gets deleted, and "what did I believe in May vs.
now" must always be answerable. That means every fact needs bi-temporal validity
(`valid_at` / `invalid_at`), automatic supersession when a new fact contradicts an
old one, and hybrid retrieval (semantic + keyword + graph traversal) over the
result.

Two build options:

1. **Raw Neo4j.** Hand-model the temporal schema, write the Cypher for supersession
   detection, build the embedding pipeline, and implement hybrid search (vector
   index + full-text + BFS + a reranker) ourselves.
2. **Graphiti** (Apache-2.0, the engine behind Zep). It ships exactly this: an
   LLM-driven extraction pipeline that turns text episodes into typed entities and
   edges, native bi-temporal edges, automatic contradiction/supersession handling,
   and a hybrid searcher with reciprocal-rank fusion over Neo4j.

## Decision

Build on Graphiti (`graphiti-core`, pinned to 0.29.1), backed by self-hosted Neo4j
Community. Synapse wraps Graphiti in a thin `KnowledgeEngine`; our custom schema is
expressed as Graphiti custom entity/edge types rather than a bespoke graph model.
The temporal model is *native* — we map our `valid_from`/`valid_until` onto
Graphiti's edge-level bi-temporal fields and do not re-implement them.

## Consequences

**Positive.** The hardest parts — LLM extraction, supersession, and hybrid ranking —
are solved and battle-tested by the engine's authors. We inherit a maintained
pipeline and can spend our effort on retrieval quality, curation safety, and the
agent interface. Pinning to a proven paper's stack (the Zep temporal-KG paper) is
the lowest-risk path for a one-person build.

**Negative — the real trade-off.** We are coupled to Graphiti's model and its gaps.
Two bit us concretely. Graphiti does *not* create a vector index on the
relationship embeddings it writes, so per-write dedup was doing a full cosine scan
until we added our own `synapse_relates_fact_vec` index. And Graphiti's default
reranker returns *rank positions*, not similarity scores — a fact we only discovered
while tuning retrieval (see ADR-0004's sibling work), forcing us to compute real
cosine per result ourselves. Coupling to a young library (0.29.x) means version
churn: the two-tier `model`/`small_model` config and the OpenAI-generic client path
are version-specific footguns. We accept this coupling because reimplementing a
correct temporal-KG engine solo would cost far more than working around its edges.
