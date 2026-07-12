# Write Pipeline

Plan Part 4. Implemented in `synapse/core/write_pipeline.py`. How knowledge gets in.

## Flow

```
remember(content, type?, project_id?, agent_role?, source?, force?)
   │
   1+2. CAPTURE + TRIAGE   ── Haiku ── worth storing? type? global?      (R2 filter)
   │        └─ not worth storing & not forced ─────────────────► REJECTED
   3.  SCOPE               ── agent_Y > global signal > project_X (default)
   4.  DEDUPLICATE         ── embed (BGE-M3) → nearest fact by cosine in scope+global
   │        ├─ score ≥ 0.90 ──────────────────────────────────► DUPLICATE (skip store)
   │        └─ 0.75 ≤ score < 0.90 ── Haiku adjudicate:
   │               ├─ duplicate ──────────────────────────────► DUPLICATE (skip store)
   │               ├─ contradiction ── flag, then store new ──► CONTRADICTION (+ supersede)
   │               └─ distinct ───────────────────────────────► (continue)
   5+6. SCORE + STORE      ── Graphiti add_episode (Sonnet extract + BGE-M3 embed + edges)
            └──────────────────────────────────────────────────► STORED
```

## The design principle: wrap Graphiti, don't reimplement it

Graphiti's `add_episode` already does entity extraction, embedding, and edge
creation. This pipeline adds only what Graphiti lacks:

| Step | Who does it | Why |
|------|-------------|-----|
| Write-trigger filter (R2) | **Haiku** (`ClaudeTriage`) | Graphiti stores anything; we must reject noise. Cheap, high-volume, simple classification — the safe Haiku use case. |
| Scope resolution | pipeline logic | maps to `group_id` (`global` / `project_X` / `agent_Y`) |
| Pre-store dedup | **BGE-M3** + Neo4j cosine | catch near-duplicate *episodes* before they create graph noise |
| Contradiction adjudication | **Haiku** | vector sim alone can't tell "same fact" from "opposite fact" — both are similar; needs a judgment |
| Extraction + embedding + edges | **Sonnet** via Graphiti | quality-critical, structured — the locked extraction model |

This is also how the Haiku-vs-Sonnet tension resolves: **Haiku triages, Sonnet
extracts.** Both honored, each on the job it's suited for.

## Dedup math

Incoming content is embedded (BGE-M3, 1024-dim) and compared to existing
`RELATES_TO.fact_embedding` vectors in `[target_scope, global]` via Neo4j's
`vector.similarity.cosine`. Thresholds (in `config.py`):

- `dedup_threshold = 0.90` → hard duplicate, skip storing (plan's ">0.9 → update").
- `relate_floor = 0.75` → gray zone, hand to Haiku adjudication.
- below floor → unrelated, store as new.

*Live-proven:* a restated Kafka decision matched the stored fact at **0.936** and
was correctly caught as a duplicate.

## Outcomes (`WriteResult.outcome`)

`STORED` · `DUPLICATE` (with `duplicate_of`) · `CONTRADICTION` (with `contradicts`,
new truth still stored so Graphiti's temporal model supersedes) · `REJECTED`
(failed the write-trigger filter).

## Scope rules

`agent_role` → `agent_<role>`; else global signal (or no project context) →
`global`; else → `project_<id>`. Dedup runs against the target scope **and**
`global`, so a project write that merely restates a global fact is still caught.

## Deliberate deviations from the literal Part-4 spec (flagged for review)

- **No separate Qdrant write.** The spec says "STORE → Graphiti **+ embed to
  Qdrant**." Graphiti already embeds (into Neo4j's native vector index), and that
  index is what the dedup query uses. Writing the same vectors to Qdrant now would
  build the redundancy we flagged in `embedder-provider.md`/§6C. **Decision: store
  via Graphiti only; revisit Qdrant at Phase 3** when retrieval decides whether a
  separate vector layer earns its keep. The `db/vector_client.py` seam stays empty
  until then.
- **"Update existing" on duplicate = skip, not merge.** A `>=0.9` match returns the
  existing id and does not re-store (prevents noise). A richer merge/refresh
  (bump confidence, add provenance) is a curation-engine concern (Phase 10).
- **Contradiction = flag + store new.** We record `contradicts` in the result and
  store the new fact (Graphiti invalidates the old). Persisting an explicit
  `Contradicts` edge and surfacing it in the Curate UI is Phase 9/curation.

## Testability

All external calls (triage LLM, embedder, vector index, graph) are injected
Protocols. `tests/test_write_pipeline.py` (11 tests, all green) fakes them to test
the *logic*; `scripts/write_smoke.py` runs the *real* services end-to-end.

## Known observation (retrieval-tuning item, not a bug)

Without strong cues, Sonnet sometimes extracts whole statements as `entity`
nodes (e.g. "Decision to use Kafka... due to durable replay"). The typed
attributes still populate correctly; this is normal Graphiti behavior and a
candidate for extraction-prompt tuning in Phase 3.
