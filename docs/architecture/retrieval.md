# Retrieval Engine

Plan Part 5. Implemented in `synapse/core/retrieval_engine.py`. How knowledge comes out.

## Two entry points

- **`recall(query, project_id?, agent_role?, limit?, as_of?, center_node_uuid?)`** →
  ranked `Recalled` facts. The MCP `recall`/`search` tools ride this.
- **`brief(project_id)`** → structured `Brief` (the killer feature). Redis-cached.

Both are exposed through the `KnowledgeEngine` facade (`synapse/core/knowledge_engine.py`).

## Multi-strategy retrieval

| Strategy | How |
|----------|-----|
| Semantic | Graphiti `search` → cosine similarity over **Neo4j's vector index** (BGE-M3 fact embeddings) |
| Keyword | Graphiti `search` → BM25 full-text (same call) |
| Graph traversal | Graphiti `search(center_node_uuid=…)` → BFS around a node ("what relates to this") |
| Temporal | `temporal_filter(facts, as_of)` → keep facts where `valid_at ≤ as_of < invalid_at` |

> **Why not Qdrant?** The write pipeline stores embeddings *through Graphiti into
> Neo4j*, and Graphiti's hybrid search already covers semantic + keyword + graph.
> A separate Qdrant would hold a duplicate copy of the same vectors for no added
> capability. This **confirms §6C: Qdrant is redundant for the MVP.** It stays in
> the stack unused, pending your call (see "Open decision" below). `synapse/db/
> vector_client.py` remains an empty seam.

## Ranking algorithm (tunable)

Candidates come back in Graphiti's relevance order; Synapse re-ranks with a weighted
composite (`score_facts`, `RankWeights`):

```
score = w_rel·relevance + w_rec·recency + w_conf·confidence + w_conn·connectivity   (normalized)

relevance     = 1 − rank_index/N           (position in Graphiti's hybrid result)
recency       = 0.5 ^ (age_days / half_life)   half_life default 30d
confidence    = fact/node confidence attr, else 0.5
connectivity  = node degree / max degree among candidates   (graph centrality)
```

Defaults: relevance .45 / recency .20 / confidence .20 / connectivity .15. All
tunable per call via `RankWeights`. *Live example:* a recall returned the most
on-topic, most-connected Kafka fact at 0.900 with its component breakdown.

## Scope composition (R5)

`Scope.compose(project_id, agent_role)` → `["global", "project_<id>", "agent_<role>"]`.
A Acme-Store query gets `global + project_acme-store` merged and ranked together;
`recall`/`brief` pass these as Graphiti `group_ids`.

## The `brief` operation

`brief(project_id)` runs structured **category queries** (not search) by node label
within `["global", "project_<id>"]`, and assembles:

- `project_summary` — synthesized counts (will deepen once Project nodes carry summaries)
- `active_conventions` — `Convention` nodes
- `key_decisions` — `Decision` nodes
- `relevant_lessons` — `Lesson` nodes, **sorted by severity** (critical → low)
- `cross_project_knowledge` — `Pattern`/`Decision`/`Convention` in `global` scope

### Caching (Redis)
Briefs are cached at `brief:<project_id>` (TTL 300s) in Redis (port 6382). The
`KnowledgeEngine` **busts that cache on every real write** to the project, so a
`remember()` makes the next `brief()` rebuild. Live-confirmed: first call
`cached=False`, second `cached=True`.

## Temporal handling & a known limit

`temporal_filter` correctly answers "as of date X" for facts whose validity window
spans X (unit-tested: a fact superseded 5 days ago is still returned for an `as_of`
20 days ago). **Limit:** Graphiti's `search` may exclude already-expired edges from
its candidate set, so deep-historical recall of long-superseded facts may need a
direct driver query later. Current-state and recent-history work today.

## Testability

`Searcher` + `GraphQueries` + Redis are injected Protocols. `tests/
test_retrieval_engine.py` (11 tests) covers temporal filtering, ranking signals,
scope composition, recall, brief structure, and cache use/bust — all without live
services. `scripts/retrieve_smoke.py` proves the real path end-to-end.

## Observation (retrieval-tuning, Phase 3+)

`Decision` summaries sometimes concatenate several extracted statements into one
node. Recall still ranks well; tightening the extraction prompt / splitting nodes
is a future tuning item.
