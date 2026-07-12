# Graph Explorer (Phase 7) — the centerpiece

The living, force-directed knowledge graph (plan Part 9, Screen 1). The single hero of
the UI. Built on `react-force-graph-2d` / `-3d` over the `/api/v1/graph` endpoint.

## Data
`useGraph(scopes, asOf?)` → `{ nodes, links }` (already force-graph-shaped: nodes have
`id`, links have `source`/`target` ids). Default scopes = `global` + every project; the
scope selector narrows. Type toggles filter client-side (instant, no refetch).

## Rendering
- **Node:** glowing disc, **color by type** (`nodeColor`), **radius by degree**
  (`2 + √degree·1.6`), soft `shadowBlur` glow. Label drawn when zoomed in or hovered.
  Selected node gets a cyan ring.
- **Links:** dim hairlines within a scope; **cross-project edges** (project→different
  project) highlighted cyan with directional particles — the "magic" connections.
- **Clustering:** projects self-cluster naturally (dense intra-scope links, sparse
  cross-scope); cross-project links are longer. Color = type, cluster = project.
- **2D and 3D** modes (3D via `react-force-graph-3d` + three) — toggle in controls.
- Transparent background so the atmosphere shows through.

## Interaction
- **Hover** → label + pointer.
- **Click node** → `<NodeDetailPanel>` slides in from the right (Framer Motion) with the
  node, its extracted attributes, and connected facts (in/out), from `/graph/node/{id}`.
  Actions stubbed: Edit / Supersede / Promote (wired in later phases).
- **Controls** (`<GraphControls>`, floating panel): type toggles (the legend doubles as
  filter), scope selector (All / global / each project), 2D·3D, a time-slider (`as_of`)
  + "show superseded" for temporal scrubbing.

## Real-time growth (R: aliveness)
The WebSocket store's `addedCount` invalidates the `graph` query on `knowledge.added`, so a
`remember` from any connected agent makes the node appear live. (A pulse-on-arrival polish
pass can follow.)

## State
`lib/graphStore.ts` (Zustand): `scopeMode`, `enabledTypes`, `mode` (2d/3d), `asOf`,
`includeSuperseded`, `selectedNodeId`.

## Gate (R6)
The "want to explore" test: opening `/graph` should make you want to dive in — a glowing,
clustered galaxy of your real knowledge that responds to hover/click/filter. Verified via
Playwright screenshots + manual QA.
