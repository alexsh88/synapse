# Synapse Knowledge Schema

Reference for the knowledge model. Implemented in `synapse/core/schema.py`, encoded
as **Graphiti custom types**. Read before write-pipeline / retrieval / MCP work.

## Entity types (custom labels)

Graphiti auto-provides `uuid`, `name`, `summary`, `group_id`, `created_at`, and
embeddings on every node. The classes below add **only extra attributes**; field
descriptions are extraction prompts (Claude reads them to decide what to pull).

| Label | Extra attributes | Notes |
|-------|------------------|-------|
| `Project` | tech_stack, status(active/paused/archived) | the projects in the registry |
| `Decision` | rationale, alternatives_considered, confidence(tentative/settled/locked), made_by | the "why" |
| `Convention` | category(code_style/architecture/naming/workflow/testing), example | "always do X" |
| `Lesson` | context, lesson_type(gotcha/best_practice/anti_pattern/failure), severity(low→critical), source_project | learned the hard way |
| `Research` | findings, sources, relevance_tags | concluded investigations |
| `Pattern` | problem, solution, used_in | cross-project reusable shapes |
| `Tool` | category, purpose, chosen_over, rationale | library/service choices |

Anything that doesn't fit falls back to Graphiti's generic `Entity` label (the
plan's "ENTITY generic") — no custom class needed.

## Edge types

`Supersedes`(reason), `AppliesTo`, `DerivedFrom`, `DiscoveredIn`, `Informed`,
`UsedIn`, `HasGotcha`, `Contradicts`(resolution_status), `RelatedTo`,
`References`, `SharesPattern`.

`EDGE_TYPE_MAP` constrains which edges are allowed between which labels (e.g.
`(Decision, Decision) → [Supersedes, Contradicts]`, `(Project, Project) →
[SharesPattern]` — the cross-project magic link). Graphiti also still extracts
free-form fact edges alongside the typed ones (observed: `USES`, `REPLACES`) —
that's expected and additive, not a bug.

## Temporal model — native to Graphiti (not re-implemented)

`valid_from` / `valid_until` from plan Part 3 map to Graphiti's **edge-level
bi-temporal fields** (`valid_at` / `invalid_at`). Supersession happens
automatically when a new fact contradicts an existing one — we do **not** add
temporal fields to entities. "What did Acme-Store use in June vs now" is answered
by Graphiti's `search(..., )` at a reference time. (R4)

## Scope model — maps to `group_id` (R5)

Scope is the **partition a piece of knowledge lives in**, expressed as the
episode's `group_id`. Graphiti group_ids allow only `[A-Za-z0-9_-]` — **no
colons** — so the format is underscore-joined:

| Scope | group_id |
|-------|----------|
| global | `global` |
| project | `project_<id>`  e.g. `project_acme-store` |
| agent | `agent_<role>`  e.g. `agent_planner` |

Retrieval **composes** scopes by passing several group_ids:
`Scope.compose("acme-store")` → `["global", "project_acme-store"]`. A node is
written under exactly one scope; promotion (project → global) is a re-scope, not
a copy. Helpers: `synapse.core.schema.Scope`.

## How it's wired into Graphiti

`add_episode(..., entity_types=ENTITY_TYPES, edge_types=EDGE_TYPES,
edge_type_map=EDGE_TYPE_MAP, group_id=Scope.project(...))`. Validated by
`scripts/schema_smoke.py` — typed extraction + attributes + scoped retrieval all
confirmed against Claude Sonnet 4.6 + Neo4j + BGE-M3.
