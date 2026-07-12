# Architecture Decision Records

This directory records the significant, hard-to-reverse decisions behind Synapse — a
self-hosted temporal knowledge-graph memory shared across AI coding agents. Each
record follows the [Nygard ADR format](https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions):
**Status · Context · Decision · Consequences**. They document *why*, including the
alternatives weighed and the downsides accepted — not just the final choice.

For the broader engineering narrative — what these decisions felt like in practice,
what broke, and how we know the system works — see [`../ENGINEERING.md`](../ENGINEERING.md).

## Records

| # | Decision | Status |
|---|----------|--------|
| [0001](0001-graphiti-over-raw-neo4j.md) | Use Graphiti as the temporal graph engine, not raw Neo4j | Accepted |
| [0002](0002-local-bge-m3-embeddings.md) | Local BGE-M3 embeddings via Ollama, not a hosted embedder | Accepted |
| [0003](0003-neo4j-native-vectors-over-qdrant.md) | Neo4j's native vector index, not a separate Qdrant | Accepted (reverses earlier decision) |
| [0004](0004-temporal-supersede-model.md) | Knowledge supersedes, it is never deleted | Accepted |
| [0005](0005-mcp-as-agent-interface.md) | MCP server as the agent interface | Accepted |
| [0006](0006-extraction-mode-routing.md) | Mode-configurable extraction: cloud / local / hybrid | Accepted |

## Conventions

- One decision per file, numbered sequentially (`NNNN-short-slug.md`).
- Records are immutable once accepted. To change a decision, add a new record that
  supersedes it and update this index — the same supersede-don't-delete principle the
  system itself is built on (ADR-0004).
- Keep each record to ~250–450 words: enough for the reasoning, not a design spec.
