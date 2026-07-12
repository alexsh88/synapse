# Curation Engine (Phase 10)

The background maintenance that keeps the brain healthy — dedup, staleness, and
contradiction surfacing — **without ever losing valid knowledge (R8)**. Read this
before touching `synapse/core/curation_engine.py`, `backup.py`, or `workers/`.

## The safety contract (R8, R4, R3)

Four invariants. Every operation and test upholds all four.

1. **Never hard-delete.** No `DELETE` of fact edges or entity nodes, ever. Curation
   only *flags* or *temporally supersedes*. The graph is append-and-mark, not remove (R3/R4).
2. **Every mutation is reversible.** Merge/archive set properties (`invalid_at`,
   `merged_into`, `archived`) that a later `restore()` can clear. Nothing destroys data.
3. **Backup before any mutation.** A mutating call snapshots the affected facts to
   `backups/curation-<utc-iso>.json` first, and records the path on the result. The
   periodic analysis scans are read-only and skip this.
4. **Suggestion-first, human-approved apply.** Celery tasks only *analyse* and surface
   candidates; they never auto-merge/auto-archive. Destructive intent always flows
   through an explicit API/UI `apply` call. The engine never corrupts the graph on a timer.

A regression that violated (1) — e.g. a future "real delete" — is caught by
`BackupService.verify_no_loss`: every fact uuid present before an operation must still
resolve afterward (invalidated/archived is fine; *gone* is not).

## Data model recap

Knowledge facts are `RELATES_TO` **edges** (`uuid`, `fact`, `fact_embedding`, `group_id`,
`valid_at`, `invalid_at`, optional `archived`/`merged_into`). Entities are nodes. Temporal
end = `invalid_at` set (this is exactly what `KnowledgeEngine.forget` does). Scope = `group_id`.

## Operations

### Analysis (read-only, no backup, Celery-scheduled)
- **`find_duplicates(scopes?)`** — pairs of *active* fact edges in the same scope with cosine
  similarity ≥ **`curation_dedup_threshold` (0.97)**. Union-find into clusters; the earliest-created
  edge is the suggested `canonical`, the rest are merge candidates.
  - **Why 0.97, not the write-time 0.90:** measured on the live ~589-node graph, fact↔fact at 0.90
    yields 178 clusters with many *false positives* — distinct facts sharing sentence structure score
    ~0.90 ("X computes Liquidity Sweeps" vs "X computes Order Blocks"; "uses ComfyUI for image-gen" vs
    "uses FFmpeg for video-assembly"). By 0.97 (~27 clusters) the false positives wash out. The write
    pipeline's 0.90 is for *episode→fact* (whole blurb vs fact) and is unchanged.
- **`find_stale(older_than_days=180)`** — active, non-archived facts created before the
  cutoff. Candidates to archive (NOT delete). On a freshly-seeded graph this is empty — honest.
- **`find_review_pairs(scopes?)`** — band pairs (similarity in `[curation_review_floor,
  curation_dedup_threshold)`, i.e. **0.90–0.97**) in the same scope: *possibly* related/overlapping,
  worth a human glance — we do **not** assert they conflict, and there is **no** merge button on them.
  Per-pair Haiku `adjudicate` (reusing `WritePipeline`) on demand is a future enhancement.
- **Pair-scan cap:** the O(n²) similarity scan is bounded by `curation_pair_limit` (500) and **logs a
  warning when truncated** — no silent caps. (The earlier hard-coded `LIMIT 100` silently hid ~6× the
  pairs on the 589-node graph.)
- **`suggestions(scopes?)`** — bundles the three above for `GET /curation/suggestions` → Curate UI.

### Mutations (reversible, backup-first, human-triggered only)
- **`merge_duplicate(canonical_uuid, duplicate_uuid)`** — backup, then mark the duplicate
  superseded: `invalid_at = now`, `merged_into = canonical`, `curation_reason`. The fact is
  retained (temporal end, R4); `recall(as_of=before)` still finds it. Reversible via `restore`.
- **`archive(edge_uuid)`** — backup, then `archived = true`, `archived_at = now`. A reversible
  hide-flag. Archived facts are **excluded from retrieval**: `GraphService.snapshot` (graph viz)
  and `GraphitiSearcher.search` (recall/search) both filter `coalesce(archived,false)=false`.
- **`restore(edge_uuid)`** — clear `archived`/`merged_into` and (for an archive) the flag,
  re-activating the fact. Proves reversibility; also the manual "undo" for a bad merge.

Each mutation returns `ApplyResult{ ok, edge_uuid, backup_path, action }`.

## Celery topology
`workers/celery_app.py` — Celery app on the Redis broker (`settings.redis_url`, db 6382).
`workers/curation_tasks.py` — `scan_suggestions` + `scan_health`, registered on a beat
schedule (daily). Tasks construct a short-lived `KnowledgeEngine`, run the **read-only**
analysis, and cache the result in Redis for the API/UI. No task performs a mutation.

## Testing (the gate)
`tests/test_backup.py` + `tests/test_curation_engine.py`, fake-driver style (no live Neo4j):
- merge → the duplicate edge still exists with `invalid_at` set (zero-loss).
- archive → restore round-trips the flag.
- `verify_no_loss` raises when a uuid present in the backup is missing afterward.
- mutations call `BackupService.snapshot` before issuing the write.
Gate (per plan): curation runs on the real graph and a backup diff shows **zero fact loss**.
