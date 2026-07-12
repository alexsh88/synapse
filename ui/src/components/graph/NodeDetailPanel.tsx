import { AnimatePresence, motion } from "framer-motion";
import { X } from "lucide-react";

import { useNode } from "../../lib/api";
import { useGraphStore } from "../../lib/graphStore";
import { nodeColor } from "../../lib/nodeColors";

export default function NodeDetailPanel() {
  const selectedId = useGraphStore((s) => s.selectedNodeId);
  const select = useGraphStore((s) => s.select);
  const { data, isLoading } = useNode(selectedId);

  return (
    <AnimatePresence>
      {selectedId && (
        <motion.aside
          initial={{ x: 372, opacity: 0 }}
          animate={{ x: 0, opacity: 1 }}
          exit={{ x: 372, opacity: 0 }}
          transition={{ type: "spring", stiffness: 320, damping: 34 }}
          className="absolute right-0 top-0 z-30 flex h-full w-[360px] flex-col border-l border-line bg-surface/90 backdrop-blur-md"
        >
          <header className="flex items-center justify-between border-b border-line px-4 py-3">
            <span
              className="font-mono text-[11px] uppercase tracking-wider"
              style={{ color: nodeColor(data?.node.type ?? "entity") }}
            >
              {data?.node.type ?? "…"}
            </span>
            <button onClick={() => select(null)} className="text-muted transition-colors hover:text-text">
              <X size={16} />
            </button>
          </header>

          <div className="min-h-0 flex-1 overflow-y-auto px-4 py-4">
            {isLoading && <p className="text-sm text-muted">Loading…</p>}
            {data && (
              <>
                <h2 className="font-display text-lg leading-snug text-text">{data.node.name}</h2>
                {data.node.summary && (
                  <p className="mt-2 text-sm leading-relaxed text-muted">{data.node.summary}</p>
                )}
                <div className="mt-3 flex flex-wrap gap-2 font-mono text-[11px] text-muted">
                  <span className="rounded border border-line bg-surface-2 px-2 py-0.5">{data.node.scope}</span>
                  <span className="rounded border border-line bg-surface-2 px-2 py-0.5">{data.node.degree} links</span>
                </div>

                {Object.keys(data.attributes).length > 0 && (
                  <dl className="mt-4 space-y-2.5">
                    {Object.entries(data.attributes).map(([k, v]) => (
                      <div key={k}>
                        <dt className="font-mono text-[10px] uppercase tracking-wide text-muted/70">
                          {k.replace(/_/g, " ")}
                        </dt>
                        <dd className="text-sm text-text">{String(v)}</dd>
                      </div>
                    ))}
                  </dl>
                )}

                {(data.edges_out.length > 0 || data.edges_in.length > 0) && (
                  <div className="mt-5">
                    <p className="font-mono text-[10px] uppercase tracking-wide text-muted/70">Connected</p>
                    <ul className="mt-2 space-y-2">
                      {data.edges_out.map((e, i) => (
                        <li key={`o${i}`} className="text-sm leading-snug text-muted">
                          <span className="text-accent/70">→ </span>
                          {e.fact || e.name}
                        </li>
                      ))}
                      {data.edges_in.map((e, i) => (
                        <li key={`i${i}`} className="text-sm leading-snug text-muted">
                          <span className="text-accent/70">← </span>
                          {e.fact || e.name}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </>
            )}
          </div>

          <footer className="flex gap-2 border-t border-line px-4 py-3">
            {["Edit", "Supersede"].map((a) => (
              <button
                key={a}
                disabled
                title="Coming in a later phase"
                className="rounded-lg border border-line bg-surface-2 px-3 py-1.5 text-xs text-muted opacity-40 cursor-not-allowed"
              >
                {a}
              </button>
            ))}
            <button
              disabled
              title="Coming in a later phase"
              className="rounded-lg border border-accent/30 bg-accent/10 px-3 py-1.5 text-xs text-accent opacity-40 cursor-not-allowed"
            >
              Promote
            </button>
          </footer>
        </motion.aside>
      )}
    </AnimatePresence>
  );
}
