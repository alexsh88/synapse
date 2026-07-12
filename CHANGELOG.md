# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
