import { useRef, useState } from "react";
import { ChevronDown } from "lucide-react";

import { useGraph, useProjects } from "../../lib/api";
import { useGraphStore } from "../../lib/graphStore";
import { KNOWLEDGE_TYPES, NODE_COLORS } from "../../lib/nodeColors";
import { cn } from "../../lib/cn";

// ---------------------------------------------------------------------------
// Themed custom dropdown (replaces the native <select>)
// ---------------------------------------------------------------------------
interface DropdownOption { value: string; label: string }

function ThemedDropdown({
  value,
  options,
  onChange,
}: {
  value: string;
  options: DropdownOption[];
  onChange: (v: string) => void;
}) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  const current = options.find((o) => o.value === value);

  // Close on outside click
  const onBlur = () => setTimeout(() => setOpen(false), 120);

  return (
    <div ref={ref} className="relative mt-1 w-full" onBlur={onBlur}>
      <button
        type="button"
        onClick={() => setOpen((p) => !p)}
        className={cn(
          "flex w-full items-center justify-between rounded-lg border border-line bg-surface-2 px-2 py-1.5 text-text",
          "focus:outline-none transition-colors",
          open && "border-accent/40 ring-1 ring-accent/20",
        )}
      >
        <span className="truncate text-xs">{current?.label ?? value}</span>
        <ChevronDown
          size={12}
          className={cn("ml-1 shrink-0 text-muted transition-transform", open && "rotate-180")}
        />
      </button>

      {open && (
        <div className="absolute left-0 top-full z-50 mt-1 w-full overflow-hidden rounded-lg border border-line bg-surface shadow-[0_8px_32px_rgba(0,0,0,0.5)]">
          {options.map((o) => (
            <button
              key={o.value}
              type="button"
              onMouseDown={() => { onChange(o.value); setOpen(false); }}
              className={cn(
                "block w-full px-2.5 py-1.5 text-left text-xs transition-colors hover:bg-surface-2",
                o.value === value ? "text-accent" : "text-text",
              )}
            >
              {o.label}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// GraphControls
// ---------------------------------------------------------------------------
export default function GraphControls() {
  const { data: projects } = useProjects();
  const {
    scopeMode, setScopeMode, enabledTypes, toggleType,
    mode, setMode, includeSuperseded, setIncludeSuperseded,
    asOf, setAsOf,
  } = useGraphStore();

  // Use the current graph data (for the same scopes) to find the earliest node timestamp
  const scopes =
    scopeMode === "all"
      ? ["global", ...(projects?.map((p) => `project_${p.id}`) ?? [])]
      : [scopeMode];
  const { data: graphData } = useGraph(scopes);

  const now = Date.now();
  const fallbackEarliest = now - 90 * 24 * 60 * 60 * 1000; // 90 days ago

  const earliestMs = (() => {
    if (!graphData?.nodes.length) return fallbackEarliest;
    let min = Infinity;
    for (const n of graphData.nodes) {
      // GraphNode doesn't carry created_at in the type, but the API may send it
      const raw = (n as unknown as Record<string, unknown>)["created_at"];
      if (typeof raw === "string") {
        const ms = Date.parse(raw);
        if (!isNaN(ms) && ms < min) min = ms;
      }
    }
    return min === Infinity ? fallbackEarliest : min;
  })();

  // Slider position: 0–100 maps from earliestMs to now
  const asOfMs = asOf ? Date.parse(asOf) : now;
  const span = now - earliestMs;
  const sliderVal =
    span === 0
      ? 100
      : Math.round(((Math.min(Math.max(asOfMs, earliestMs), now) - earliestMs) / span) * 100);

  const [dragging, setDragging] = useState(false);

  const handleSlider = (v: number) => {
    const ms = earliestMs + (v / 100) * (now - earliestMs);
    setAsOf(new Date(ms).toISOString());
  };

  const isLive = asOf === null;

  // Scope dropdown options
  const scopeOptions: DropdownOption[] = [
    { value: "all", label: "All projects" },
    { value: "global", label: "Global" },
    ...(projects?.map((p) => ({ value: `project_${p.id}`, label: p.name })) ?? []),
  ];

  return (
    <div className="absolute left-4 top-4 z-20 w-52 rounded-xl border border-line/70 bg-surface/70 p-3 text-xs backdrop-blur-md">
      <label className="font-mono text-[10px] uppercase tracking-wide text-muted/70">Scope</label>
      <ThemedDropdown
        value={scopeMode}
        options={scopeOptions}
        onChange={setScopeMode}
      />

      {/* ---- Time slider ---- */}
      <p className="mt-3 font-mono text-[10px] uppercase tracking-wide text-muted/70">Time</p>
      <div className="mt-1.5 space-y-1.5">
        <div className="flex items-center gap-1.5">
          <input
            type="range"
            min={0}
            max={100}
            value={isLive ? 100 : sliderVal}
            onMouseDown={() => setDragging(true)}
            onMouseUp={() => setDragging(false)}
            onTouchStart={() => setDragging(true)}
            onTouchEnd={() => setDragging(false)}
            onChange={(e) => {
              const v = Number(e.target.value);
              if (v === 100) setAsOf(null);
              else handleSlider(v);
            }}
            className="w-full accent-cyan-400 cursor-pointer"
            style={{ accentColor: "#22d3ee" }}
          />
          <button
            onClick={() => setAsOf(null)}
            className={cn(
              "shrink-0 rounded-md border px-1.5 py-0.5 font-mono text-[9px] uppercase tracking-wider transition-colors",
              isLive
                ? "border-accent/40 bg-accent/10 text-accent"
                : "border-line bg-surface-2 text-muted hover:text-text",
            )}
          >
            Live
          </button>
        </div>
        {(dragging || !isLive) && (
          <p className="text-center font-mono text-[10px] text-accent/80">
            {isLive
              ? "Now"
              : new Date(earliestMs + (sliderVal / 100) * (now - earliestMs)).toLocaleDateString(
                  undefined,
                  { month: "short", day: "numeric", year: "numeric" },
                )}
          </p>
        )}
      </div>

      <p className="mt-3 font-mono text-[10px] uppercase tracking-wide text-muted/70">Types</p>
      <div className="mt-1.5 space-y-0.5">
        {KNOWLEDGE_TYPES.map((t) => (
          <button
            key={t}
            onClick={() => toggleType(t)}
            className={cn(
              "flex w-full items-center gap-2 rounded px-1.5 py-1 text-left capitalize transition-colors hover:bg-surface-2",
              !enabledTypes.has(t) && "opacity-35",
            )}
          >
            <span
              className="h-2.5 w-2.5 rounded-full"
              style={{ background: NODE_COLORS[t], boxShadow: `0 0 6px ${NODE_COLORS[t]}` }}
            />
            <span className="text-text">{t}</span>
          </button>
        ))}
      </div>

      <div className="mt-3 flex items-center gap-1 rounded-lg border border-line bg-surface-2 p-0.5">
        {(["2d", "3d"] as const).map((m) => (
          <button
            key={m}
            onClick={() => setMode(m)}
            className={cn(
              "flex-1 rounded-md py-1 font-mono uppercase transition-colors",
              mode === m ? "bg-accent/15 text-accent" : "text-muted hover:text-text",
            )}
          >
            {m}
          </button>
        ))}
      </div>

      <label className="mt-2.5 flex cursor-pointer items-center gap-2 text-muted">
        <input
          type="checkbox"
          checked={includeSuperseded}
          onChange={(e) => setIncludeSuperseded(e.target.checked)}
          className="accent-cyan-400"
        />
        Show superseded
      </label>
    </div>
  );
}
