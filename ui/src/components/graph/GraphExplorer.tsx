import { Suspense, lazy, useEffect, useMemo, useRef, useState } from "react";
import ForceGraph2D from "react-force-graph-2d";
import { useQueryClient } from "@tanstack/react-query";

import { useGraph, useProjects } from "../../lib/api";
import { useGraphStore } from "../../lib/graphStore";
import { nodeColor } from "../../lib/nodeColors";
import { useWs, getNewNodeTimestamps } from "../../lib/ws";

// react-force-graph's prop surface is huge; cast to relax strict typing for the viz props.
const FG2D = ForceGraph2D as unknown as React.ComponentType<Record<string, unknown>>;
// 3D (three.js) is code-split — only loaded when the user switches to 3D mode.
const FG3D = lazy(() => import("react-force-graph-3d")) as unknown as React.ComponentType<Record<string, unknown>>;

const isXProject = (l: { source: unknown; target: unknown }) => {
  const s = (typeof l.source === "object" && l.source !== null && "scope" in l.source
    ? (l.source as { scope?: string }).scope
    : "") ?? "";
  const t = (typeof l.target === "object" && l.target !== null && "scope" in l.target
    ? (l.target as { scope?: string }).scope
    : "") ?? "";
  return s !== t && s.startsWith("project_") && t.startsWith("project_");
};

export default function GraphExplorer() {
  const containerRef = useRef<HTMLDivElement>(null);
  const fgRef = useRef<{ zoomToFit?: (ms: number, padding: number) => void } | null>(null);
  const fitDataRef = useRef<unknown>(null);   // graphData last auto-fit to (fit on data change only)
  const [dims, setDims] = useState({ w: 800, h: 600 });
  const [hoverId, setHoverId] = useState<string | null>(null);

  const { data: projects } = useProjects();
  const { scopeMode, enabledTypes, mode, asOf, includeSuperseded, selectedNodeId, select } = useGraphStore();
  const addedCount = useWs((s) => s.addedCount);
  const qc = useQueryClient();

  const scopes = useMemo(
    () => (scopeMode === "all" ? ["global", ...(projects?.map((p) => `project_${p.id}`) ?? [])] : [scopeMode]),
    [scopeMode, projects],
  );
  const { data } = useGraph(scopes, { asOf: asOf ?? undefined, includeSuperseded });

  // real-time: a knowledge.added event refetches the graph so it grows live.
  useEffect(() => {
    if (addedCount) qc.invalidateQueries({ queryKey: ["graph"] });
  }, [addedCount, qc]);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const measure = () => setDims({ w: el.clientWidth, h: el.clientHeight });
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    measure();
    return () => ro.disconnect();
  }, []);

  const graphData = useMemo(() => {
    if (!data) return { nodes: [], links: [] };
    const nodes = data.nodes.filter((n) => enabledTypes.has(n.type));
    const ids = new Set(nodes.map((n) => n.id));
    const links = data.links.filter(
      (l) => ids.has(l.source as string) && ids.has(l.target as string),
    );
    // clone — force-graph mutates x/y/source/target; don't poison the query cache.
    return { nodes: nodes.map((n) => ({ ...n })), links: links.map((l) => ({ ...l })) };
  }, [data, enabledTypes]);

  // Memoized so hover (which only flips hoverId for the label) does NOT recreate these props.
  // Recreating them churned react-force-graph and reheated the simulation, and onEngineStop
  // then re-fit the view — so every hover zoomed all the way out. Deps exclude hoverId.
  const common = useMemo(
    () => ({
      ref: fgRef,
      width: dims.w,
      height: dims.h,
      graphData,
      backgroundColor: "rgba(0,0,0,0)",
      nodeRelSize: 4,
      nodeVal: (n: { degree?: number }) => 1 + (n.degree || 0),
      cooldownTicks: 120,
      onNodeClick: (n: { id: string }) => select(n.id),
      onNodeHover: (n: { id?: string } | null) => setHoverId(n?.id ?? null),
      onBackgroundClick: () => select(null),
      // Fit only when the DATA changes (initial load, filter/scope change, real-time growth) —
      // never on a hover-induced settle. Zoom follows data, not the cursor.
      onEngineStop: () => {
        if (fitDataRef.current !== graphData) {
          fgRef.current?.zoomToFit?.(500, 60);
          fitDataRef.current = graphData;
        }
      },
      linkColor: (l: { source: unknown; target: unknown }) =>
        isXProject(l) ? "rgba(34,211,238,0.6)" : "rgba(120,140,170,0.16)",
      linkWidth: (l: { source: unknown; target: unknown }) => (isXProject(l) ? 1.4 : 0.5),
      linkDirectionalParticles: (l: { source: unknown; target: unknown }) => (isXProject(l) ? 2 : 0),
      linkDirectionalParticleWidth: 1.8,
      linkDirectionalParticleColor: () => "#22d3ee",
    }),
    [graphData, dims.w, dims.h, select],
  );

  if (mode === "3d") {
    return (
      <div ref={containerRef} className="absolute inset-0">
        <Suspense fallback={<div className="grid h-full place-items-center text-sm text-muted">Loading 3D…</div>}>
          <FG3D
            {...(common as Record<string, unknown>)}
            nodeColor={(n: { type: string }) => nodeColor(n.type)}
            nodeOpacity={0.92}
          />
        </Suspense>
      </div>
    );
  }

  return (
    <div ref={containerRef} className="absolute inset-0">
      <FG2D
        {...(common as Record<string, unknown>)}
        nodeCanvasObject={(node: { x?: number; y?: number; id?: string; type?: string; degree?: number; name?: string }, ctx: CanvasRenderingContext2D, scale: number) => {
          if (!Number.isFinite(node.x) || !Number.isFinite(node.y)) return;  // skip transient NaN coords
          const nx = node.x!;
          const ny = node.y!;
          const r = 2.6 + Math.sqrt(node.degree || 1) * 2.1;
          const color = nodeColor(node.type ?? "entity");

          // New-node entry pulse: decaying outer halo for nodes received via WS in last 2s.
          // Pruning now happens via lazy interval in ws.ts.
          if (node.id) {
            const ts = getNewNodeTimestamps().get(node.id);
            if (ts !== undefined) {
              const age = Date.now() - ts;
              const t = Math.max(0, 1 - age / 2000);           // 1→0 over 2 s
              const ringR = r * 3.4 + (1 - t) * r * 4;         // expanding ring
              const alpha = t * 0.7;
              ctx.save();
              ctx.strokeStyle = color;
              ctx.globalAlpha = alpha;
              ctx.shadowColor = color;
              ctx.shadowBlur = 60 * t;
              ctx.lineWidth = 2 / scale;
              ctx.beginPath();
              ctx.arc(nx, ny, ringR, 0, 2 * Math.PI);
              ctx.stroke();
              ctx.restore();
            }
          }

          // soft outer halo — makes nodes read as glowing stars
          const halo = ctx.createRadialGradient(nx, ny, r * 0.4, nx, ny, r * 3.4);
          halo.addColorStop(0, `${color}55`);
          halo.addColorStop(1, "rgba(0,0,0,0)");
          ctx.fillStyle = halo;
          ctx.beginPath();
          ctx.arc(nx, ny, r * 3.4, 0, 2 * Math.PI);
          ctx.fill();
          // glowing core
          ctx.shadowColor = color;
          ctx.shadowBlur = 16;
          ctx.beginPath();
          ctx.arc(nx, ny, r, 0, 2 * Math.PI);
          ctx.fillStyle = color;
          ctx.fill();
          ctx.shadowBlur = 0;
          if (node.id === selectedNodeId) {
            ctx.strokeStyle = "#22d3ee";
            ctx.lineWidth = 1.5 / scale;
            ctx.beginPath();
            ctx.arc(nx, ny, r + 3 / scale, 0, 2 * Math.PI);
            ctx.stroke();
          }
          if (scale > 1.7 || node.id === hoverId || node.id === selectedNodeId) {
            const name = node.name ?? "";
            const label = name.length > 36 ? `${name.slice(0, 34)}…` : name;
            ctx.font = `${11 / scale}px 'Inter Variable', 'Inter', sans-serif`;
            ctx.textAlign = "center";
            ctx.textBaseline = "top";
            ctx.fillStyle = "rgba(230,232,235,0.82)";
            ctx.fillText(label, nx, ny + r + 2.5 / scale);
          }
        }}
        nodePointerAreaPaint={(node: { x?: number; y?: number; degree?: number }, color: string, ctx: CanvasRenderingContext2D) => {
          if (!Number.isFinite(node.x) || !Number.isFinite(node.y)) return;
          const r = 2.6 + Math.sqrt(node.degree || 1) * 2.1 + 3;
          ctx.fillStyle = color;
          ctx.beginPath();
          ctx.arc(node.x!, node.y!, r, 0, 2 * Math.PI);
          ctx.fill();
        }}
      />
    </div>
  );
}
