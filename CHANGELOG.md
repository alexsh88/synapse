# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-08-19

### Added
- Runbooks — procedural memory as ordered, executable steps, exposed as the `remember_runbook` and
  `runbooks` MCP tools (bringing the tool count to nine)
- Re-embedding migration: script plus runbook, for the one-way door the 1024-dim index represents
- Credential and identifier redaction on the write path, including brokerage account identifiers,
  with a retroactive corpus-redaction script for values stored before the rule existed
- Multi-root project registry (`EXTRA_PROJECT_ROOTS`) — a project may live under more than one root

### Changed
- Write provenance records a named host instead of a container id, which is ephemeral and useless
  for the one query the field exists to answer
- ruff excludes `.claude/` so `ruff check .` and the scoped run agree

### Fixed
- Graphiti's automatic edge invalidation is no longer trusted blindly — it was expiring facts it had
  no business expiring
- One scope parser shared by the HTTP and MCP interfaces, so clusters resolve identically in both
- Repaired 9 dead MCP wirings, and stopped connected projects shadowing the `synapse` package
- Hybrid extraction now escalates an *empty* local extraction, not only a failed one
- Project connect: the graph write moved out of the request path; the job contract validates
- Deep-seed reads the target project's own docs instead of Synapse's boilerplate
- Connector wiring written from inside the container pointed back into the container
- The venv interpreter is derived from the host rather than hardcoded to Windows, so the same repo
  wires correctly from a Windows and a macOS host

## [0.1.0] - 2026-07-11

### Added
- Hybrid local/cloud extraction routing for cost optimization (local gemma3:12b with Sonnet fallback)
- Automatic session-lesson capture on PreCompact/SessionEnd hooks
- Retrieval evaluation harness with baseline regression gate (hit@k 0.562, MRR 0.484)
- Negative cases and precision@k metrics for retrieval assessment
- Native vector index support for embeddings storage without separate vector DB
- Hash dedup and shared credit state for write pipeline
- Atomic update and verify_no_loss safety checks for graph curation
- K-NN curation and scoped backup strategies
- Global error handler and contract hardening on 16 API routes
- Bounded event streaming and optional authentication on API
- Real similarity ranking and temporal back-fill for retrieval
- Validity-filtered brief curation with temporal superseding
- Dockerization of UI and integration into compose stack
- Real cosine similarity scores with min_relevance floor (retrieval violations reduced 3→0, MRR improved .484→.531)
- Time slider for temporal scrubbing in graph explorer
- Entry animations, transitions, and lazy loading in UI
- Queue-and-replay writes when local embedder (Ollama) is unavailable
- Credit-aware fallback from Anthropic API to local extraction

### Changed
- Response models now explicit on 16 API routes with proper typing
- Extraction mode configurable: cloud (Sonnet), local (gemma3 + json_schema), or hybrid
- Improved retrieval ranking with real-time cosine scores vs RRF
- UI performance optimizations: pulse map pruning on interval, vendor chunk splitting
- Graph curation now prevents cross-project metric miscounts (shared concepts vs cross-scope edges)
- Removed unused dependencies and updated coverage configuration

### Fixed
- Graph hover behavior no longer zooms to extremes
- UI-added projects now properly displayed in project list
- Retrieval violations eliminated through tighter min_relevance thresholds
- Concurrent write safety ensured through atomic graph operations
