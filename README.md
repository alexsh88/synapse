# Synapse

**A self-hosted, temporal knowledge-graph memory for AI coding agents** — shared across projects, queryable across time, written and read via MCP by any Claude Code session.

---

## What makes Synapse different

Most agent memory systems store a flat collection of text chunks with embeddings and retrieve by cosine similarity. That answers "what do I know?" — it cannot answer "what did I think in May, and when did that change?"

Synapse builds a **temporal knowledge graph** where every fact has a `valid_from` and `invalid_at` timestamp. When newer evidence contradicts old knowledge, the old node is **superseded, not deleted** — the full history of what was known and when is always queryable. This is the model Graphiti (the engine behind Zep) uses, and it changes the class of questions you can answer.

It also differs from raw graph databases: Graphiti handles entity resolution, temporal deduplication, and hybrid vector+graph retrieval out of the box. Synapse adds MCP tooling, multi-scope isolation (global / project / agent), cross-project knowledge linking, extraction-mode routing with credit-aware fallback, and a retrieval eval harness with a regression gate.

[![CI](https://github.com/alexsh88/synapse/actions/workflows/ci.yml/badge.svg)](https://github.com/alexsh88/synapse/actions/workflows/ci.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

---

![Temporal scrubbing in the Graph Explorer — the graph re-rendered at a past date, then blooming back to Live](docs/images/synapse-demo.gif)

*The time slider re-renders the graph as it existed at any past date — then "Live" blooms it back to the present (~2,900 nodes).*

---

## Features

- **Temporal knowledge model** — every fact carries `valid_from` / `invalid_at`; knowledge is superseded, never deleted. "What did the agent believe in May?" is always answerable.
- **MCP tools for any Claude Code session** — `remember`, `recall`, `brief`, `relate`, `search`, `forget`, `update` wired into any project with a single JSON stanza.
- **`brief(project_id)` session-start context** — loads the distilled knowledge snapshot for a project at the start of every coding session; cached in Redis, served in milliseconds.
- **Hybrid extraction routing with credit-aware fallback** — `cloud` (Claude Sonnet 4.6), `local` (gemma3:12b via Ollama, $0), or `hybrid` (local → Sonnet on failure). When local embedding is unavailable, writes are queued and replayed rather than dropped silently.
- **Retrieval eval harness with regression gate** — golden set with positive, negative, and cross-project-leakage cases; MRR 0.531, 0 violations on the author's corpus. A >5% relative drop in hit@k or MRR, or any increase in violations, exits non-zero and blocks the run.
- **Fail-closed degraded-path design** — extraction failures go to a review queue; the write pipeline never silently discards knowledge. Health endpoints surface degraded state so operators know.
- **Curation with verify-no-loss safety** — dedup, decay, and archival tasks are gated by a `verify_no_loss` check that asserts no valid knowledge was removed before committing.

---

## Architecture

```mermaid
graph LR
    subgraph "Agent / Claude Code"
        A[Claude Code session]
    end

    subgraph "MCP Layer"
        M[MCP Server<br/>remember · recall · brief<br/>relate · search · forget · update]
    end

    subgraph "Synapse API  :8848"
        F[FastAPI]
        C[Celery workers]
        WS[WebSocket hub]
    end

    subgraph "Storage"
        N[(Neo4j Community<br/>graph + 1024-dim<br/>BGE-M3 vector index)]
        R[(Redis<br/>brief cache +<br/>Celery broker)]
    end

    subgraph "Local ML  Ollama"
        E[BGE-M3<br/>embedder]
        G[gemma3:12b<br/>local extraction]
    end

    subgraph "Cloud LLM  optional"
        S[Claude Sonnet 4.6<br/>cloud extraction]
    end

    subgraph "React UI  :5174"
        UI[Graph Explorer 2D/3D<br/>Timeline · Search<br/>Curate · Documents]
    end

    A -- "MCP stdio" --> M
    M --> F
    F --> N
    F --> R
    C --> N
    C --> R
    F -- "extraction: hybrid" --> G
    G -. "fallback" .-> S
    F --> E
    E --> N
    WS -- "graph growth events" --> UI
    F --> WS
    UI --> F
```

---

## Quickstart

**Prerequisites**

- Docker and Docker Compose
- [Ollama](https://ollama.ai) on the host with BGE-M3 pulled:
  ```bash
  ollama pull bge-m3
  ```
  _Or_ skip Ollama entirely and use cloud extraction instead (see the `.env` note below).

**Run**

```bash
git clone https://github.com/alexsh88/synapse.git
cd synapse

cp .env.example .env
# Open .env and set NEO4J_PASSWORD to any strong password.
# For cloud-only mode (no Ollama), also set:
#   EXTRACTION_MODE=cloud
#   ANTHROPIC_API_KEY=sk-ant-...

docker compose up -d
# Wait ~60 s for Neo4j to finish its first start.
# All four services become (healthy): neo4j, redis, api, ui.

open http://localhost:5174   # Graph Explorer UI
# API:          http://localhost:8848
# Neo4j Browser: http://localhost:7475  (user: neo4j, password: <your NEO4J_PASSWORD>)
```

**Wire a Claude Code project** (after the stack is healthy):

```bash
# 1. Register the project: copy projects.example.json to projects.json and add
#    an entry with your project's id, name, and path.
# 2. Wire it (writes .mcp.json + hook stubs into the target project):
python -m scripts.wire_project my-project
```

This writes `.mcp.json` and hook stubs into the target project. From then on every Claude Code session in that project can call `remember`, `recall`, and `brief`.

---

## Design decisions

The hard calls — graph engine, vector strategy, embedder, extraction routing — each have an ADR that records the trade-offs considered and the option rejected:

| ADR | Decision |
|-----|----------|
| [0001](docs/decisions/0001-graphiti-over-raw-neo4j.md) | Graphiti over raw Neo4j (entity resolution, temporal model, hybrid retrieval) |
| [0002](docs/decisions/0002-local-bge-m3-embeddings.md) | Local BGE-M3 via Ollama (zero per-token cost, the embedder Graphiti's paper used) |
| [0003](docs/decisions/0003-neo4j-native-vectors-over-qdrant.md) | Neo4j native vector index over Qdrant (redundant at this scale; one store) |
| [0004](docs/decisions/0004-temporal-supersede-model.md) | Temporal supersede model (valid_from/invalid_at; history always queryable) |
| [0005](docs/decisions/0005-mcp-as-agent-interface.md) | MCP as the agent interface (native Claude Code integration, zero wrapper code) |
| [0006](docs/decisions/0006-extraction-mode-routing.md) | Extraction mode routing (cloud/local/hybrid with credit-aware fallback) |

Deeper engineering notes live in [docs/ENGINEERING.md](docs/ENGINEERING.md).

---

## Testing

**201 tests** across unit, integration, and temporal-invariant suites.

The temporal-invariant tests (`tests/test_temporal_invariants.py`) assert on the Cypher queries emitted to Neo4j — verifying that `valid_from` / `invalid_at` constraints are written and enforced correctly, not just that the Python layer behaves.

```bash
# Full test suite
pytest tests/ -v

# Core engines only (fast, no live Neo4j needed)
pytest tests/test_curation_engine.py tests/test_retrieval_engine.py tests/test_graph_service.py -v
```

**Retrieval eval harness**

The eval harness runs the golden set against the live engine and scores hit@k, MRR, precision@k, and violations:

```bash
# Run against the live engine
python -m scripts.run_eval

# Save current run as the new baseline (writes synapse/eval/baseline.json, gitignored)
python -m scripts.run_eval --save-baseline

# Every subsequent run is the regression gate: exits non-zero if hit@k or MRR
# drops >5% relative, or if violations increase vs. the saved baseline.
python -m scripts.run_eval
```

On the author's production corpus (~2,900 nodes across ten projects) the gated baseline sits at MRR **0.531**, hit@k **0.562**, **0 violations**. A demo case set ships in `synapse/eval/cases.py`; point it at your own corpus and save your own baseline.

![Synapse UI in Docker](docs/images/synapse-ui-docker.png)

---

## Project structure

```
synapse/
├── synapse/
│   ├── core/
│   │   ├── knowledge_engine.py    # Graphiti wrapper
│   │   ├── write_pipeline.py      # extract → scope → dedup → store
│   │   ├── retrieval_engine.py    # multi-strategy search + ranking
│   │   ├── curation_engine.py     # background maintenance
│   │   └── schema.py              # entity / relationship definitions
│   ├── mcp/
│   │   ├── server.py              # MCP server entry point
│   │   └── tools.py               # remember, recall, brief, relate, search, forget, update
│   ├── api/
│   │   ├── main.py                # FastAPI app
│   │   ├── routes/                # knowledge, graph, search, curation, timeline, projects
│   │   └── websocket.py           # real-time graph-growth events
│   ├── workers/
│   │   └── curation_tasks.py      # Celery tasks (dedup, decay, archival)
│   ├── eval/
│   │   ├── cases.py               # EvalCase definitions — the golden set
│   │   └── runner.py              # scoring + baseline regression gate
│   ├── models/                    # Pydantic request/response schemas
│   ├── db/
│   │   ├── neo4j_client.py
│   │   └── vector_client.py
│   └── config.py                  # pydantic-settings; all env vars documented here
├── ui/
│   └── src/
│       ├── components/
│       │   ├── ui/                # shadcn/ui primitives (unmodified)
│       │   └── domain/            # Synapse components (GraphExplorer, NodeDetail, …)
│       ├── pages/                 # GraphPage, TimelinePage, SearchPage, CuratePage, …
│       ├── hooks/
│       ├── lib/                   # API client, WebSocket, Zustand stores
│       └── types/
├── tests/                         # 201 tests (unit + integration + temporal-invariant)
├── scripts/                       # run_eval.py, wire_project.py, seed helpers
├── docs/
│   ├── architecture/              # schema, write-pipeline, retrieval, mcp-server, api
│   ├── decisions/                 # ADRs 0001–0006
│   └── research/                  # embedder, vector store, extraction model comparisons
├── docker-compose.yml
├── Dockerfile
├── .env.example
└── requirements.txt
```

---

## Known limitations

- **Single-node Neo4j Community** — no clustering or horizontal read scaling. Adequate for the intended workload (tens of connected projects, thousands of knowledge nodes); re-evaluate at graph sizes above ~500k nodes.
- **Embedding dimension locked at first ingestion** — BGE-M3 produces 1024-dimensional vectors; this is set in the Neo4j vector index schema on first write. Changing the dimension requires dropping the index and re-embedding the entire corpus.
- **Local extraction drops ~14% of dense facts** — gemma3:12b in `local` mode silently misses roughly 14% of complex, information-dense writes compared to Claude Sonnet 4.6. These are detected by the write pipeline's structural validator and moved to the review queue rather than dropped silently, but they do require manual review.
- **Desktop-first UI** — the graph explorer requires a real screen to be useful. Mobile layout does not break, but it is not the design target.
- **Eval golden set is small** — ~15 cases covering positive, negative, and cross-project categories. Meaningful for regression detection; not large enough for statistical confidence in absolute metric comparisons. Extend `synapse/eval/cases.py` as real usage accumulates.

---

## AI attribution

Built with Claude Code as a development accelerator — architecture, design decisions, trade-off analysis, and testing strategy are the author's.

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
