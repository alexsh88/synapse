import { Command } from "lucide-react";
import { useGraph, useProjects } from "../../lib/api";
import { usePalette } from "../../lib/commandStore";
import LiveDot from "../ui/LiveDot";

export default function TopBar() {
  const { data: projects } = useProjects();
  const { data: globalGraph } = useGraph(["global"]);

  const openPalette = usePalette((s) => s.setOpen);
  const projectNodes = projects?.reduce((sum, p) => sum + p.nodes, 0) ?? 0;
  // Derive total from fetched data only — addedCount is excluded to avoid drift
  // (the graph query already refetches on knowledge.added events).
  const total = projectNodes + (globalGraph?.nodes.length ?? 0);

  return (
    <header className="z-20 flex h-14 shrink-0 items-center gap-4 border-b border-line/70 bg-surface/30 px-5 backdrop-blur-sm">
      <div className="flex items-center gap-2.5 select-none">
        <span className="h-2 w-2 rounded-full bg-accent shadow-[0_0_10px_2px_var(--color-accent)]" />
        <span className="font-display text-[19px] font-medium tracking-tight text-text">Synapse</span>
      </div>

      <button
        onClick={() => openPalette(true)}
        className="group ml-2 flex items-center gap-2 rounded-lg border border-line bg-surface-2/60 px-3 py-1.5 text-sm text-muted transition-colors hover:border-muted/40 hover:text-text"
      >
        <Command size={13} strokeWidth={1.75} />
        <span>Search the brain…</span>
        <kbd className="ml-2 rounded border border-line bg-bg px-1.5 py-0.5 font-mono text-[10px] text-muted">⌘K</kbd>
      </button>

      <div className="ml-auto flex items-center gap-4 font-mono text-xs text-muted">
        <span>
          <span className="text-text">{total.toLocaleString()}</span> nodes
        </span>
        <span className="flex items-center gap-2">
          <LiveDot />
          <span className="hidden sm:inline">live</span>
        </span>
      </div>
    </header>
  );
}
