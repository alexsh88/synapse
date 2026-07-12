# MCP Server

Plan Part 6. The agent interface to Synapse. `synapse/mcp/server.py` (FastMCP, stdio)
+ `synapse/mcp/tools.py` (tool logic over `KnowledgeEngine`).

## Tools (7)

| Tool | Maps to | Notes |
|------|---------|-------|
| `remember(content, type?, scope?, relationships?)` | `engine.remember` | write-trigger filtered; scope="global" or default project |
| `recall(query, scope?, limit?, as_of?)` | `engine.recall` | global+project, ranked; `as_of` ISO for point-in-time |
| `brief(project_id?)` | `engine.brief` | the killer feature; Redis-cached |
| `search(query, filters?)` | `engine.search` | across ALL projects; filters: scope/limit/as_of (type/confidence reported in `filters_ignored`) |
| `relate(from_id, to_id, relationship_type)` | `engine.relate` | manual structural edge (no embedding → not in semantic search) |
| `forget(knowledge_id, reason?)` | `engine.forget` | temporal end (invalid_at=now), NOT deletion (R4) |
| `update(knowledge_id, changes)` | `engine.update` | supersede: invalidate old + store new (R4) |

## Architecture

- **FastMCP** over **stdio**. One `KnowledgeEngine` connected for the process lifetime
  via the FastMCP `lifespan` (builds indices, wires write + retrieval), closed on shutdown.
- **Scope resolution:** `SYNAPSE_PROJECT_ID` env var → the default project for
  `remember`/`recall`/`brief`/`update`. `scope="global"` overrides to global.
- **Error handling:** every tool body runs through `_safe()` — exceptions become
  `{"error": "..."}` (logged to stderr) instead of dropping the MCP connection.
- **Logging to stderr only** (stdout is the MCP protocol channel; never `print()`).
- `tools.py` is engine-agnostic (pure functions) → unit-tested with a fake engine.

## Connecting a project

Copy `.mcp.json.example` into the project's `.mcp.json`, set `SYNAPSE_PROJECT_ID`.
The server is launched with Synapse's venv python and `PYTHONPATH`; it reads
Neo4j/Ollama/Anthropic config from Synapse's **own `.env`** (resolved by absolute
path in `config.py`), so no secrets live in the connected project.

In CLAUDE.md, instruct the agent: call `brief` at session start; `remember` on
decisions/conventions/lessons/research; `recall`/`search` before solving a problem.

## Tests

- `tests/test_mcp_tools.py` (13): scope resolution, param mapping, result shapes,
  error paths, and that the server registers exactly the 7 tools. All green (suite 35/35).
- `scripts/mcp_smoke.py`: lists tools (inspector view) and exercises all 7 live.

## ✅ Resolved: embedder NaN (Ollama flash-attention bug)

**Symptom:** bge-m3 on Ollama intermittently returned NaN embeddings →
`Ollama 500: "json: unsupported value: NaN"`, failing the write. It worsened over a
session of rapid `add_episode` runs.

**Root cause (researched — `docs/research/ollama-bge-m3-nan.md`):** a known Ollama bug
where bge-m3 (and other embedders) produce NaN via a **flash-attention numerical
instability** on certain inputs (Ollama issues #13572, #14657, #9639, #12921). The 500
is a secondary symptom — Go's JSON can't encode NaN. It's input-sensitive *and*
flash-attention-related (hence the on-the-edge intermittency).

**Fix (applied + validated):** set **`OLLAMA_FLASH_ATTENTION=0`** for the Ollama server
(the documented root-cause workaround) and restart Ollama. The exact string that failed
15/15 before now passes 0/20; the full live 7-tool smoke runs clean (0 embed failures).
The env var is persisted via `setx` so it survives restarts.
> Durability note: ensure the Ollama service/app starts with `OLLAMA_FLASH_ATTENTION=0`
> (persisted in the user environment). Also worth updating Ollama to a build with PR
> #14739, which makes any residual NaN fail gracefully instead of a cryptic 500.

**Defense-in-depth still in place** (`knowledge_engine.py`): `RetryingOpenAIEmbedder`
(retry + per-item batch fallback; never zero-vectors — invalid for cosine/dedup, so a
persistent failure raises and surfaces as a tool `error`) and a `graphiti_max_coroutines`
cap. If NaN ever recurs, the fallback option is a different local embedder
(`nomic-embed-text`, 768-dim — changes the LOCKED 1024 dim, requires re-embedding).
