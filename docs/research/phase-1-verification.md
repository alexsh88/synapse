# Phase 1A — Architecture Re-Verification

**Date:** 2026-05-30
**Method:** Web research (the agent-memory space changes monthly; plan Part 0 Mandatory Research Protocol).
**Verdict:** ✅ **Stack is stable — PROCEED with the planned core (Graphiti + Neo4j + Qdrant + react-force-graph).** No locked decision is invalidated. But **three adjustments** should be made before Phase 2 (one touches a CLAUDE.md convention). Details in §6.

---

## 1. Is Graphiti still the best self-hostable temporal knowledge graph?

**Yes — unchanged leader for the temporal niche. Keep it.**

- Actively maintained: **v0.29.1, released 2026-05-21**, ~26.7k stars. Not stagnant.
- Still the only mainstream option where **time is first-class**: facts get `valid_at`/`invalid_at` validity windows and are superseded, not deleted — exactly our R4 ("temporal model is sacred"). Bi-temporal (records both when a fact became true and when the system learned it).
- Benchmarks still favor the temporal-graph approach for time/relationship queries (Zep/Graphiti **63.8%** vs Mem0 **49.0%** on LongMemEval/GPT-4o; the temporal validity-window design drives the ~15pt gap).

**Competitors checked:**
- **Mem0** — vector-first, optional *thin* graph (entity→rel→entity, no NL facts, **no temporal model**); graph features now gated behind a **$249/mo Pro tier** for self-host. Improved on temporal queries in Apr 2026 but still not a true temporal graph. Not a fit for our temporal requirement.
- **Cognee** — strongest *fully air-gapped* option (no external calls), graph-first, supports Neo4j/FalkorDB/Kuzu/Qdrant. Less explicit temporal focus than Graphiti. Worth knowing as a fallback if Graphiti's OpenAI-embedding coupling (§6B) ever becomes a blocker, but it trades away the temporal depth that is our whole point.
- No new entrant displaces Graphiti for *temporal* knowledge graphs.

> Caveat (already true in our plan): self-hosting the **full Zep platform** is no longer supported — only the **Graphiti engine** is. We only ever planned to use the engine, so no impact. Migration path to managed Zep Cloud remains open (plan Part 13 intact).

## 2. Neo4j vs FalkorDB vs Kuzu for self-hosted Graphiti

**Keep Neo4j as default. This call aged well — and one alternative just died.**

Graphiti's current supported backends: **Neo4j 5.26+** · **FalkorDB 1.1.2+** · **Kuzu 0.11.2+** · Amazon Neptune.

- **Neo4j (our choice)** — default, best-documented, production-proven. Our `docker-compose.yml` pins `neo4j:5-community` (latest 5.x ≫ 5.26 ✓) with APOC, which Graphiti needs. ✅
- **⚠️ Kuzu is now DEPRECATED/archived** (continues as a "LadybugDB" fork). The plan listed Kuzu as a possible lighter alternative — **strike it.** Picking Neo4j over Kuzu was the right call; using Kuzu now would be a maintenance-risk trap.
- **FalkorDB** — the *correct* lighter alternative going forward: Redis-module, low-latency, GraphRAG-focused, **speaks Bolt** so it's a near drop-in for Neo4j drivers. Source-available license (check fit before adopting). **→ Update Risk #2 mitigation: "FalkorDB as the lighter fallback," not Kuzu.**

> Reminder from the research: backend choice matters *less* for cost/latency than the LLM/embedding provider — every episode ingestion fires multiple LLM calls. See §6.

## 3. Has Anthropic shipped native cross-project memory? (Issue #36561)

**No — not as a real cross-project, searchable, temporal store. Synapse's differentiation is intact. Scope unchanged. Risk #1 stays MED.**

What exists today is still file-based and flat:
- `CLAUDE.md` (project + global `~/.claude/CLAUDE.md`) — hand-maintained, loaded at session start.
- **Auto-memory** — but strictly **per-project**; not shared across repos.
- The **official "Memory MCP"** — a knowledge graph stored as a **JSONL file** (`./.claude/memory.json` or global `~/.claude/memory.json`), entities/relations/observations. This is the closest official thing to cross-project memory, but it is **flat, file-backed, non-temporal, no semantic ranking, no UI** — i.e. exactly the file-backed approach our R3 rejects, and missing temporal (R4), multi-scope composition (R5), retrieval ranking (R7), and the beautiful UI (R6).

Synapse still does materially more: temporal validity windows, multi-scope composition, ranked retrieval, curation, and the graph UI. **No scope change required.**

## 4. Is react-force-graph still the best graph viz for React?

**Still a valid primary choice — keep it, especially for the 3D "galaxy" mode. One new alternative to evaluate at Phase 7.**

- **react-force-graph** — WebGL, React-native, 2D **and 3D** (Three.js). The 3D mode is still the standout for a "living brain" aesthetic (R6). Force-directed only, excellent perf. No reason to drop it.
- **Reagraph** (new, gaining traction) — WebGL, React-first (`<GraphCanvas nodes edges />`), GPU-accelerated, 2D/3D. Cleaner React API than react-force-graph. **Action: bake-off Reagraph vs react-force-graph at the start of Phase 7** before committing the centerpiece. Not urgent now.
- Cytoscape.js (rich algorithms, heavier integration), React Flow (already our secondary for editable/structured views — confirmed correct), ECharts (only if we also need many chart types). No change to the plan's primary/secondary split.

## 5. Any new MCP memory server that already does what we're building?

**Several overlap in parts; none combines all of Synapse's pillars (temporal graph + multi-scope + ranked retrieval + curation + beautiful UI). Build still justified — but study the closest prior art.**

- **doobidoo/mcp-memory-service** ⭐ closest — self-hosted, REST API + MCP + OAuth + CLI + **dashboard**, knowledge graph, **autonomous consolidation**, ~5ms retrieval, **Remote MCP** for native claude.ai. **→ Action: read its design before building our MCP server (Phase 4) and curation engine — strongest source of "don't reinvent" lessons, and its Remote-MCP support is worth copying for multi-machine access.**
- **mem0-mcp-selfhosted** — fully local (Qdrant + Ollama embeddings + optional Neo4j), 11 tools, semantic cross-project memory. Confirms our self-hosted stack shape is mainstream; lacks temporal validity windows.
- **shaneholloman/mcp-knowledge-graph** — fork of the official server; flat entities/relations, synced-folder for cross-machine. Non-temporal.
- **viralvoodoo Claude Code Memory Server** — Neo4j-backed, tracks decisions/patterns across sessions. Non-temporal, no UI.
- **DeusData/codebase-memory-mcp** — structural *code* indexing (155 langs, sub-ms), not conversational/decision knowledge. Complementary, not competitive.

**Conclusion:** the unique Synapse combination — Graphiti temporal graph + global/project/agent scope composition + ranked multi-strategy retrieval + `brief()` + the graph UI — is not offered by any single existing server. Adopt **patterns** from doobidoo (Remote MCP, consolidation) rather than the whole thing.

---

## 6. Required Adjustments Before Phase 2 (the part that shifted)

### A. ⚠️ Extraction model: "Haiku for extraction" is risky as written — verify or switch to Sonnet
CLAUDE.md §4 says *"Use Claude Haiku for entity extraction."* Graphiti's pipeline is **structured-output-critical** (it parses strict JSON schemas for entities/edges/dedup), and its own docs warn: *"works best with services that support Structured Output... smaller models may not output the correct JSON structures and cause ingestion failures."*
- Anthropic shipped **constrained-decoding structured outputs** (beta, `anthropic-beta: structured-outputs-2025-11-13`), but at the time of the sources it was confirmed for **Sonnet 4.5+/Opus 4.1**, with **Haiku 4.5 support listed as "coming."** 2026 benchmarks: Claude Sonnet ~**99.8%** schema compliance via tool use.
- **Recommendation:** Use **Sonnet 4.6** for extraction to start (reliability > cost while we validate the MVP). In Phase 2, run a small bake-off — if **Haiku 4.5 structured outputs** is GA and passes our extraction tests, switch to Haiku for the cost win (it's the high-volume path) and update CLAUDE.md. Tune `SEMAPHORE_LIMIT` to avoid Anthropic 429s.

### B. ⚠️ Embeddings: Graphiti's Anthropic path defaults to **OpenAI embeddings** — decide our self-hosted answer (touches R9)
Graphiti separates the **LLM client** from the **embedder client**. Using **Anthropic for the LLM still expects an OpenAI key for embeddings + reranking by default.** Shipping an OpenAI dependency would violate **R9 (self-hosted, own the data)** and our "no managed deps without discussion."
- **Self-hosted-clean option (preferred):** point Graphiti's embedder at a **local OpenAI-compatible endpoint** — Ollama (`nomic-embed-text`/`mxbai-embed-large`) or a sentence-transformers server. Plan Part 7 already lists `sentence-transformers` as the local path; this just makes it the *embedder*, not an afterthought.
- **Pragmatic-start option:** accept OpenAI embeddings for the MVP only (cheap, zero-ops), with a conscious note that it breaks pure self-hosting — revisit before rollout.

> **DECISION (2026-05-30): Deferred to Phase 2.** Proceed with Phase 1 infra now (Neo4j/Qdrant/Redis don't depend on the embedder choice). The embedder provider is an **open question to resolve when wiring extraction in Phase 2** — preferred default remains the local OpenAI-compatible endpoint (Ollama / sentence-transformers) to honor R9, but OpenAI-for-MVP stays on the table.

### C. Possible simplification to revisit (not now): is Qdrant redundant?
Graphiti performs its **own** hybrid search (semantic + BM25 + graph) using embeddings it stores via the graph DB / its embedder — it doesn't strictly need an external Qdrant for *its* retrieval. Our plan adds Qdrant as a separate vector layer (LOCKED). Keep it for now (it's in the locked decisions and may serve our *own* extra semantic features), but **flag for Phase 3:** if Graphiti's built-in retrieval covers our needs, dropping Qdrant is "one fewer service." No action this phase.

### D. Minor housekeeping
- Strike **Kuzu** from "lighter alternatives" everywhere; replace with **FalkorDB** (§2).
- Add **Reagraph** to the Phase-7 viz bake-off (§4).
- Add **doobidoo/mcp-memory-service** to Phase-4 prior-art reading (§5).

---

## 7. Bottom Line

| Decision | Status | Action |
|----------|--------|--------|
| Graphiti (temporal KG engine) | ✅ Confirmed best-in-class, actively maintained | None — proceed |
| Neo4j Community (graph DB) | ✅ Confirmed (default, supported 5.26+) | None — compose is correct |
| Qdrant (vector store) | ✅ Keep (locked); possibly redundant | Revisit at Phase 3 (§6C) |
| react-force-graph (+ React Flow) | ✅ Keep | Bake-off vs Reagraph at Phase 7 |
| Self-hosted-first, migration open | ✅ Intact | None |
| Native cross-project memory shipped? | ❌ No — Synapse still differentiated | None — Risk #1 stays MED |
| **Claude Haiku for extraction** | ⚠️ **Reliability risk** | **Start with Sonnet 4.6; bake-off Haiku 4.5 in Phase 2 (§6A)** |
| **Embeddings provider** | ⚠️ **OpenAI assumed by default; R9 conflict** | **Decide: local Ollama/ST endpoint (preferred) vs OpenAI-for-MVP (§6B)** |
| Kuzu as lighter fallback | ❌ Deprecated | Use **FalkorDB** instead (§6D) |

**Recommendation: proceed to Phase 1 build. The two ⚠️ items (extraction model, embeddings provider) are Phase-2 decisions but I need your call on §6B (it's an R9 / managed-dependency question).**

---

## Sources
- [Agent Memory & Knowledge Systems Compared (2026)](https://fountaincity.tech/resources/blog/agent-memory-knowledge-systems-compared/) · [Mem0 vs Zep/Graphiti (Vectorize)](https://vectorize.io/articles/mem0-vs-zep) · [Best AI Agent Memory Frameworks 2026 (Atlan)](https://atlan.com/know/best-ai-agent-memory-frameworks-2026/) · [State of AI Agent Memory 2026 (Mem0)](https://mem0.ai/blog/state-of-ai-agent-memory-2026)
- [getzep/graphiti (GitHub)](https://github.com/getzep/graphiti) · [graphiti-core (PyPI)](https://pypi.org/project/graphiti-core/) · [Graphiti LLM Configuration (Zep docs)](https://help.getzep.com/graphiti/configuration/llm-configuration) · [Graphiti as agent memory store (Codex blog)](https://codex.danielvaughan.com/2026/03/30/graphiti-agent-memory-store/)
- [Neo4j Alternatives 2026 (ArcadeDB)](https://arcadedb.com/blog/neo4j-alternatives-in-2026-a-fair-look-at-the-open-source-options/) · [FalkorDB vs Neo4j for AI](https://www.falkordb.com/blog/falkordb-vs-neo4j-for-ai-applications/) · [Kuzu deprecation note (GitLab GKG)](https://gitlab.com/gitlab-org/rust/knowledge-graph/-/work_items/254)
- [Claude API Structured Output guide (Wiegold)](https://thomas-wiegold.com/blog/claude-api-structured-output/) · [Structured Output across providers (Glukhov, Medium)](https://medium.com/@rosgluk/structured-output-comparison-across-popular-llm-providers-openai-gemini-anthropic-mistral-and-1a5d42fa612a)
- [Claude Code memory / auto-memory (MindStudio)](https://www.mindstudio.ai/blog/what-is-claude-code-auto-memory) · [Claude Features 2026 (Suprmind)](https://suprmind.ai/hub/claude/features/)
- [React Graph Visualization Guide (Cambridge Intelligence)](https://cambridge-intelligence.com/blog/react-graph-visualization-library/) · [Reagraph (GitHub)](https://github.com/reaviz/reagraph)
- [doobidoo/mcp-memory-service (GitHub)](https://github.com/doobidoo/mcp-memory-service) · [shaneholloman/mcp-knowledge-graph (GitHub)](https://github.com/shaneholloman/mcp-knowledge-graph) · [Self-hosted mem0 MCP for Claude Code (DEV)](https://dev.to/n3rdh4ck3r/how-to-give-claude-code-persistent-memory-with-a-self-hosted-mem0-mcp-server-h68)
