import { type FormEvent, useMemo, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { Search } from "lucide-react";
import { AnimatePresence, motion } from "framer-motion";

import PageHeader from "../components/PageHeader";
import { useSearch } from "../lib/api";
import { prettyScope } from "../lib/scope";
import type { Recalled } from "../lib/types";
import ErrorBanner from "../components/ErrorBanner";

type GroupBy = "none" | "scope";
type SortBy = "relevance" | "recent";

const COMPONENT_COLOR: Record<string, string> = {
  relevance: "var(--color-accent)",
  recency: "#a78bfa",
  confidence: "#4ade80",
  connectivity: "#60a5fa",
};

const ts = (r: Recalled) => (r.valid_at ? Date.parse(r.valid_at) : 0);

export default function SearchPage() {
  const [params, setParams] = useSearchParams();
  const urlQ = params.get("q") ?? "";

  const [q, setQ] = useState(urlQ);

  const [scopeFilter, setScopeFilter] = useState<string>("all");
  const [groupBy, setGroupBy] = useState<GroupBy>("none");
  const [sortBy, setSortBy] = useState<SortBy>("relevance");

  // useSearch drives the query; the URL param is the committed search term.
  const { data: results = [], isFetching, isError } = useSearch(urlQ);

  function go(e: FormEvent) {
    e.preventDefault();
    const next = q.trim();
    if (!next) return;
    setScopeFilter("all");
    setParams(next ? { q: next } : {});
  }

  const scopes = useMemo(
    () => Array.from(new Set(results.map((r) => r.scope))).sort(),
    [results],
  );

  const view = useMemo(() => {
    let rows = scopeFilter === "all" ? results : results.filter((r) => r.scope === scopeFilter);
    if (sortBy === "recent") rows = [...rows].sort((a, b) => ts(b) - ts(a));
    return rows;
  }, [results, scopeFilter, sortBy]);

  const groups = useMemo(() => {
    if (groupBy !== "scope") return [{ key: null as string | null, rows: view }];
    const map = new Map<string, Recalled[]>();
    for (const r of view) (map.get(r.scope) ?? map.set(r.scope, []).get(r.scope)!).push(r);
    return [...map.entries()]
      .sort((a, b) => b[1].length - a[1].length)
      .map(([key, rows]) => ({ key, rows }));
  }, [view, groupBy]);

  const searched = urlQ.trim().length > 0;

  return (
    <div className="h-full overflow-y-auto px-8 py-7">
      <PageHeader title="Search" subtitle="Hybrid semantic + graph search across every project." />

      <form
        onSubmit={go}
        className="mt-5 flex max-w-2xl items-center gap-2 rounded-xl border border-line bg-surface/60 px-3.5 py-2.5 transition-colors focus-within:border-accent/40"
      >
        <Search size={16} className="text-muted" />
        <input
          value={q}
          onChange={(e) => setQ(e.target.value)}
          placeholder="Ask the brain anything…"
          className="w-full bg-transparent text-sm text-text placeholder:text-muted/60 focus:outline-none"
          autoFocus
        />
        {q && (
          <kbd className="rounded border border-line bg-bg px-1.5 py-0.5 font-mono text-[10px] text-muted">↵</kbd>
        )}
      </form>

      {isError && <ErrorBanner className="mt-4 max-w-2xl" />}

      {searched && results.length > 0 && (
        <div className="mt-4 flex max-w-2xl flex-wrap items-center gap-x-5 gap-y-2 font-mono text-[11px] text-muted">
          <Facet label="scope">
            <Chip active={scopeFilter === "all"} onClick={() => setScopeFilter("all")}>
              all · {results.length}
            </Chip>
            {scopes.map((s) => (
              <Chip key={s} active={scopeFilter === s} onClick={() => setScopeFilter(s)}>
                {prettyScope(s)} · {results.filter((r) => r.scope === s).length}
              </Chip>
            ))}
          </Facet>
          <Facet label="sort">
            <Chip active={sortBy === "relevance"} onClick={() => setSortBy("relevance")}>relevance</Chip>
            <Chip active={sortBy === "recent"} onClick={() => setSortBy("recent")}>recent</Chip>
          </Facet>
          <Facet label="group">
            <Chip active={groupBy === "none"} onClick={() => setGroupBy("none")}>none</Chip>
            <Chip active={groupBy === "scope"} onClick={() => setGroupBy("scope")}>scope</Chip>
          </Facet>
        </div>
      )}

      <div className="mt-5 max-w-2xl space-y-5 pb-12">
        {isFetching && <p className="text-sm text-muted">Searching…</p>}
        {!isFetching && searched && results.length === 0 && !isError && (
          <p className="text-sm text-muted">No matches.</p>
        )}

        {groups.map((g) => (
          <div key={g.key ?? "_all"} className="space-y-2">
            {g.key && (
              <div className="flex items-center gap-2 font-mono text-[10px] uppercase tracking-wider text-muted/70">
                <span>{prettyScope(g.key)}</span>
                <span className="h-px flex-1 bg-line/60" />
                <span>{g.rows.length}</span>
              </div>
            )}
            <AnimatePresence initial={false}>
              {g.rows.map((r, i) => (
                <motion.div
                  key={r.uuid}
                  layout
                  initial={{ opacity: 0, y: 6 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.22, delay: Math.min(i, 8) * 0.02 }}
                  className="rounded-lg border border-line/70 bg-surface/40 p-3.5 transition-colors hover:border-line"
                >
                  <p className="font-display text-[15px] leading-snug text-text">{r.fact}</p>
                  <div className="mt-2.5 flex items-center gap-3 font-mono text-[11px] text-muted">
                    <span className="text-accent">{r.score.toFixed(3)}</span>
                    <span>{prettyScope(r.scope)}</span>
                    {r.valid_at && <span className="text-muted/70">{r.valid_at.slice(0, 10)}</span>}
                    <ScoreComponents components={r.components} />
                  </div>
                </motion.div>
              ))}
            </AnimatePresence>
          </div>
        ))}
      </div>
    </div>
  );
}

function Facet({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex items-center gap-1.5">
      <span className="uppercase tracking-wider text-muted/60">{label}</span>
      {children}
    </div>
  );
}

function Chip({ active, onClick, children }: { active: boolean; onClick: () => void; children: React.ReactNode }) {
  return (
    <button
      onClick={onClick}
      className={
        "rounded-md px-2 py-0.5 transition-colors " +
        (active ? "bg-accent/15 text-accent" : "text-muted hover:text-text")
      }
    >
      {children}
    </button>
  );
}

/** Tiny stacked bars showing the relevance/recency/confidence/connectivity breakdown. */
function ScoreComponents({ components }: { components?: Record<string, number> }) {
  if (!components || Object.keys(components).length === 0) return null;
  const entries = Object.entries(components).filter(([, v]) => v > 0);
  if (entries.length === 0) return null;
  return (
    <span className="ml-auto flex items-center gap-1.5" title={entries.map(([k, v]) => `${k} ${v}`).join(" · ")}>
      {entries.map(([k, v]) => (
        <span key={k} className="h-1 w-6 overflow-hidden rounded-full bg-surface-2">
          <span
            className="block h-full rounded-full"
            style={{ width: `${Math.min(100, v * 100)}%`, background: COMPONENT_COLOR[k] ?? "var(--color-muted)" }}
          />
        </span>
      ))}
    </span>
  );
}
