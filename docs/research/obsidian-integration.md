# Research — Can Obsidian help / be part of Synapse?

**Date:** 2026-06-03 · **Question:** Should Obsidian (https://obsidian.md) be part of Synapse,
and if so in what role? **Verdict:** Not as core architecture or source of truth (R3 forbids it,
R6 requires the custom UI). **Yes** as an *optional, one-directional companion* — a read-only
**export/mirror** is the high-value, low-risk role; **capture/ingest** is a nice-to-have.

## What Obsidian is (current facts, verified 2026-06-03)
- **Local-first plain-Markdown vault.** Notes = `.md` files + YAML frontmatter on your disk. Open
  formats, you own the data, offline. Free for personal use; Sync/Publish are paid add-ons.
- **Graph view + Canvas** over vault wikilinks; thousands of community plugins/themes via an open API.
- **Bases** (core plugin, v1.9.10 Aug 2025; Bases API in v1.10.0 Oct 2025) — database-like views
  over notes, backed by frontmatter; `.base` YAML files; GUI tables/filters/formulas (a native
  Dataview successor). Still "all data stays in plain files."
- **Local REST API plugin** (coddingtonbear) — desktop-only HTTPS server (port 27124), API-key auth,
  full note CRUD, *surgical* PATCH (by heading/block/frontmatter), search (text / Dataview DQL /
  JsonLogic), command execution, and a **built-in MCP server at `/mcp/`** (Streamable HTTP).
- **Many Obsidian MCP servers exist** (jacksteamdev/obsidian-mcp-tools ~87k installs w/ semantic
  search; cyanheads; aaronsb graph-navigation). Security caveat: MCP grants broad read/write/delete
  to the vault — backups + scoping matter.

## The hard constraint: R3 (use a database, not files)
> "File-backed memory dies on concurrent writes. Neo4j transactions handle this. Never fall back to
> file-based knowledge storage."

Obsidian **is** a markdown-file store. Synapse's whole point is a temporal, multi-scope graph written
**concurrently by ~10 projects' agents**. That is precisely the concurrent-write failure mode R3 warns
about. **Therefore Obsidian must never be the source of truth or the write target of the agents.**
This isn't a close call — it's the locked decision that motivated choosing Neo4j/Graphiti.

Also: **Obsidian's graph only graphs vault wikilinks**, not a Neo4j graph, and can't do temporal
scrubbing, scope composition, or real-time WebSocket growth. So it **cannot replace** the bespoke
GraphExplorer (R6: "Obsidian graph view *but more beautiful*" — a custom, branded, temporal experience).

## Where Obsidian *can* fit (one-directional only)

### A. Read-only export / mirror  ★ recommended (high value, low risk)
Neo4j stays the source of truth. A curation/export task renders the graph → a generated vault:
one note per knowledge node (frontmatter = `scope`, `type`, `valid_from`/`valid_until`, `confidence`;
`[[wikilinks]]` = edges; folders by scope). Re-generated on a schedule, never hand-edited.
- **Gains:** offline + **mobile** reading (Obsidian mobile), a second graph view, **Bases** tables
  over knowledge, full-text search — all "for free," no UI work.
- **Doubles as a human-readable backup** (complements `BackupService`, supports the Phase-12
  "export/backup tooling + migration readiness" output).
- **Honors R3** (DB is truth; vault is a derived view), **R9** (fully local), **R6** (doesn't touch
  the hero UI). No concurrent-write risk because nothing writes back.

### B. Capture / ingest source (medium value)
Let the user braindump in Obsidian; Synapse *reads* the vault (via the Local REST API or a watcher)
and runs new/changed notes through the existing **write pipeline** (triage → store in Neo4j). Vault →
Neo4j, one-way. Lower priority because the MCP `remember` tool + per-project CLAUDE.md already cover
capture; this just adds a comfier human surface.

### C. Bidirectional sync — ✗ avoid
Two writable stores (file edits vs graph supersession) = conflict hell and the exact R3 hazard.
Not worth the complexity for a personal tool.

### D. Obsidian's own MCP servers — not a fit for *this* need
They expose *a vault* to an AI client. Synapse already exposes its *graph* via its own MCP server.
They're peers, not a stack. (Relevant only if we picked option B and chose the REST API to read the vault.)

## STATUS — option A IMPLEMENTED (2026-06-04)
Read-only export shipped: `synapse/core/obsidian_export.py` + `python -m scripts.export_obsidian
[out_dir]`. Renders the active graph to a markdown vault (one note/node, YAML frontmatter with
`type`/`scope`/`tags` for Bases, `[[wikilinks]]` for edges, folders by scope, generated README).
The exporter OWNS its output dir (wipes + rewrites). First run: 589 notes / 816 links across 11
scopes → `exports/obsidian/`. One-directional; doubles as a human-readable backup. Tests:
`tests/test_obsidian_export.py`.

## Recommendation
1. **Keep Obsidian out of the core** (no store, no agent writes, doesn't replace the UI). Lock this.
2. **Adopt option A (export/mirror) as an optional feature**, naturally slotting into Phase 12
   (export/backup/migration) — *after* Phase 11 rollout, not before. It's the cheapest way to get
   mobile + offline + a Bases/graph alt-view without diluting the bespoke UI.
3. **Treat option B as a future idea** (`docs/future-ideas.md`) — revisit if manual capture-in-Obsidian
   becomes a real habit.
4. **Do not block Phase 11 on any of this.**

## Sources
- https://obsidian.md/ · https://help.obsidian.md/bases
- https://github.com/coddingtonbear/obsidian-local-rest-api
- https://github.com/jacksteamdev/obsidian-mcp-tools · https://github.com/aaronsb/obsidian-mcp-plugin
- https://alternativeto.net/news/2025/8/obsidian-launches-new-bases-plugin-for-database-workflows-and-property-format-changes/
- https://www.neowin.net/news/obsidian-1100-released-with-new-features-and-improvements-for-bases/
