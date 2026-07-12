import { useMemo, useState } from "react";
import { Network } from "lucide-react";
import { useNavigate } from "react-router-dom";

import { useGraph, useNode, useProjects } from "../lib/api";
import { KNOWLEDGE_TYPES, nodeColor } from "../lib/nodeColors";
import { useGraphStore } from "../lib/graphStore";
import { cn } from "../lib/cn";
import type { GraphNode } from "../lib/types";

const scopeLabel = (s: string) => (s === "global" ? "Global" : s.replace("project_", ""));
const TYPES: readonly string[] = KNOWLEDGE_TYPES;

export default function DocumentsPage() {
  const { data: projects } = useProjects();
  const scopes = useMemo(
    () => ["global", ...(projects?.map((p) => `project_${p.id}`) ?? [])],
    [projects],
  );
  const { data } = useGraph(scopes);
  const [docId, setDocId] = useState<string | null>(null);
  const { data: doc } = useNode(docId);
  const navigate = useNavigate();
  const selectGraphNode = useGraphStore((s) => s.select);

  // scope -> type -> nodes (only typed knowledge counts as a "document")
  const tree = useMemo(() => {
    const m = new Map<string, Map<string, GraphNode[]>>();
    for (const n of data?.nodes ?? []) {
      if (!TYPES.includes(n.type)) continue;
      let byType = m.get(n.scope);
      if (!byType) { byType = new Map(); m.set(n.scope, byType); }
      let arr = byType.get(n.type);
      if (!arr) { arr = []; byType.set(n.type, arr); }
      arr.push(n);
    }
    return m;
  }, [data]);

  return (
    <div className="flex h-full">
      <aside className="w-64 shrink-0 overflow-y-auto border-r border-line/70 bg-surface/30 px-2 py-3">
        {scopes.map((s) => {
          const byType = tree.get(s);
          const total = byType ? [...byType.values()].reduce((a, v) => a + v.length, 0) : 0;
          if (!total) return null;
          return (
            <div key={s} className="mb-3">
              <div className="px-2 py-1 font-mono text-[10px] uppercase tracking-wide text-muted/70">
                {scopeLabel(s)} <span className="text-muted/40">· {total}</span>
              </div>
              {KNOWLEDGE_TYPES.map((t) => {
                const items = byType?.get(t);
                if (!items?.length) return null;
                return (
                  <div key={t} className="mb-1">
                    <div className="flex items-center gap-2 px-2 py-0.5 text-xs capitalize">
                      <span className="h-2 w-2 rounded-full" style={{ background: nodeColor(t) }} />
                      <span className="text-muted">{t}</span>
                      <span className="text-muted/40">{items.length}</span>
                    </div>
                    {items.map((it) => (
                      <button
                        key={it.id}
                        onClick={() => setDocId(it.id)}
                        className={cn(
                          "block w-full truncate rounded px-2 py-1 pl-6 text-left text-[13px] transition-colors hover:bg-surface-2",
                          docId === it.id ? "bg-surface-2 text-text" : "text-muted",
                        )}
                      >
                        {it.name}
                      </button>
                    ))}
                  </div>
                );
              })}
            </div>
          );
        })}
      </aside>

      <div className="min-w-0 flex-1 overflow-y-auto px-10 py-9">
        {!doc && (
          <div className="grid h-full place-items-center text-sm text-muted">
            Select a piece of knowledge to read.
          </div>
        )}
        {doc && (
          <article className="mx-auto max-w-2xl">
            <span className="font-mono text-[11px] uppercase tracking-wider" style={{ color: nodeColor(doc.node.type) }}>
              {doc.node.type}
            </span>
            <h1 className="mt-2 font-display text-3xl font-medium leading-tight text-text">{doc.node.name}</h1>
            <div className="mt-2 flex gap-3 font-mono text-xs text-muted">
              <span>{scopeLabel(doc.node.scope)}</span>
              <span>·</span>
              <span>{doc.node.degree} connections</span>
            </div>
            {doc.node.summary && (
              <p className="mt-6 font-display text-[17px] leading-relaxed text-text/90">{doc.node.summary}</p>
            )}
            {Object.keys(doc.attributes).length > 0 && (
              <dl className="mt-6 grid gap-3 rounded-xl border border-line/70 bg-surface/40 p-4 sm:grid-cols-2">
                {Object.entries(doc.attributes).map(([k, v]) => (
                  <div key={k}>
                    <dt className="font-mono text-[10px] uppercase tracking-wide text-muted/70">{k.replace(/_/g, " ")}</dt>
                    <dd className="mt-0.5 text-sm text-text">{String(v)}</dd>
                  </div>
                ))}
              </dl>
            )}
            {(doc.edges_out.length > 0 || doc.edges_in.length > 0) && (
              <div className="mt-7">
                <p className="font-mono text-[10px] uppercase tracking-wide text-muted/70">Connected knowledge</p>
                <ul className="mt-2 space-y-1.5 text-sm text-muted">
                  {doc.edges_out.map((e, i) => <li key={`o${i}`}>→ {e.fact || e.name}</li>)}
                  {doc.edges_in.map((e, i) => <li key={`i${i}`}>← {e.fact || e.name}</li>)}
                </ul>
              </div>
            )}
            <button
              onClick={() => { selectGraphNode(doc.node.id); navigate("/graph"); }}
              className="mt-8 inline-flex items-center gap-2 rounded-lg border border-accent/30 bg-accent/10 px-3 py-1.5 text-sm text-accent transition-colors hover:bg-accent/20"
            >
              <Network size={14} /> View in graph
            </button>
          </article>
        )}
      </div>
    </div>
  );
}
