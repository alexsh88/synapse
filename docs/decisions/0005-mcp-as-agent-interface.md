# 5. MCP server as the agent interface

**Status:** Accepted — 2026-06-02

## Context

Synapse's users are not humans clicking a UI — they are AI coding agents running
across ten projects, each needing to read shared memory at session start and write
back decisions, conventions, lessons, and research as work happens. The interface
question is: how does an agent talk to the knowledge graph?

Options considered:

1. **A REST API the agent calls via generated tool wrappers.** Works, but every
   connected project has to define and maintain its own HTTP tool layer, handle
   auth/URLs, and keep the tool schemas in sync with the backend.
2. **The Model Context Protocol (MCP).** A standard, tool-native protocol that
   Claude Code speaks directly. The agent discovers typed tools; the host launches
   the server as a subprocess over stdio. This is the same protocol the ecosystem
   is standardizing on, so connecting a new project is configuration, not code.

Synapse also has a FastAPI backend — but that exists for the human web UI (graph
explorer, timeline, curation panel). Conflating the two interfaces would force the
agent path through HTTP semantics it doesn't need.

## Decision

Expose the agent interface as an **MCP server** (FastMCP over stdio) with exactly
seven tools mapping to the `KnowledgeEngine`: `remember`, `recall`, `brief`,
`search`, `relate`, `forget`, `update`. `brief(project_id)` — session-start context
loading — is the headline tool and the system's reason to exist. Scope resolution is
env-driven (`SYNAPSE_PROJECT_ID` selects the project; `scope="global"` overrides).
Connecting a project is copying an `.mcp.json` and setting one env var; the server
reads Neo4j/Ollama/Anthropic config from Synapse's own `.env` by absolute path, so
no secrets live in the connected project. The REST API stays, but as the *UI's*
interface, not the agent's.

## Consequences

**Positive.** Zero per-project integration code — new projects connect by config.
The agent gets typed, discoverable tools in its native protocol. Because `tools.py`
is pure functions over the engine, the tool layer is unit-tested against a fake
engine, independent of a running Neo4j.

**Negative / accepted.** MCP is stdio-based, which shapes the operational model:
stdout is the protocol channel, so *all* logging must go to stderr and a stray
`print()` would corrupt the connection. Every tool body runs through a `_safe()`
wrapper that turns exceptions into `{"error": ...}` rather than dropping the MCP
connection — defensive by necessity, since a crash takes down the agent's whole
session. And the scope model leans on an environment variable rather than
authenticated identity, which is fine for a self-hosted single-operator brain but
would need real auth before multi-tenant exposure. We accept a protocol still
maturing in exchange for zero-friction fan-out across every project an agent touches.
