# Design System (Phase 6)

The visual language for the Synapse UI. Direction: **"an observatory for your own
thinking"** — deep space, the glowing graph as the only hero, a calm precise dark
frame around it. Calm by default, deep on demand (R6, plan Part 9).

## Typography (deviation from the locked "Inter" — see note)
- **Display / wordmark / headings / knowledge reading:** **Newsreader** (serif).
  Editorial, thoughtful — "a place for reading and thinking."
- **UI chrome** (nav, buttons, labels, metadata): **Instrument Sans** (grotesque).
- **Data / numbers / type-tags / code:** **JetBrains Mono**.
- All self-hosted via `@fontsource/*` (R9 — no external font CDN).

> **Note:** plan Part 9 said "UI: Inter." The `frontend-design` skill (mandated by R6/T10)
> explicitly flags Inter as generic. Honoring the deeper intent ("must be beautiful, not
> generic"), UI chrome uses **Instrument Sans** and display uses **Newsreader**. Palette,
> dark-hero, node colors, accent, and motion are unchanged from the locked spec.

## Color (locked)
```
--bg        #0a0c10   deep near-black, blue undertone
--surface   #13161c   lifted panels
--surface-2 #1a1e26   hovered/raised
--border    #232832   hairline
--text      #e6e8eb   soft white
--muted     #8b919a   secondary
--accent    #22d3ee   electric cyan — active, links, the "live signal"
node types: decision #f5a623 · convention #2dd4bf · lesson #f87171
            research #a78bfa · pattern #4ade80 · tool #60a5fa · entity #8b919a
```
Exposed as CSS variables + Tailwind `@theme`. Node colors also live in `lib/nodeColors.ts`
(shared with the graph in Phase 7).

## Atmosphere (not flat)
- Background: `--bg` + a faint radial cyan glow (top), + a subtle SVG **grain/noise** overlay
  at ~3% opacity for depth. A slow CSS **constellation** field stands in for the graph on the
  home placeholder until Phase 7 (already feels alive).
- Surfaces: 1px hairline borders, soft inner glow on focus, generous negative space.
- Depth via layered translucency + soft shadows, never harsh.

## Motion (Framer Motion, 200–400ms eased)
- Page transitions: quick crossfade. Panels: slide + fade. New knowledge: pulse + glow.
- Staggered reveal on first load (the skill's "one well-orchestrated page load").
- `prefers-reduced-motion` respected.

## App shell (plan Part 9)
- **NavRail** (56px, icon-only, left): Graph / Documents / Timeline / Search / Curate / Settings.
  Active item glows cyan (left bar + icon tint); hover tooltip.
- **TopBar**: ◉ Synapse wordmark (Newsreader) · ⌘K pill (center-ish) · right: live node count
  (mono) + a pulsing **live dot** reflecting WebSocket state.
- **Main**: routed view fills the rest.

## Components (this phase)
shadcn-compatible structure (`components/ui`, `cn()` util, CSS-var theme) so real shadcn
components drop in later. Hand-rolled now: `Button`, `Tooltip`, `LiveDot`, `KnowledgeCard`
(node-type colored), `NavRail`, `TopBar`, `AppShell`. Pages are placeholders wired to the API
(node count, projects) — the real screens land in Phases 7–9.

## "Want to explore" gate (R6)
Phase 6 is done when the empty shell already feels like a calm, deep place you'd want to open —
dark, atmospheric, the live dot breathing, the constellation drifting — even before the real graph.
