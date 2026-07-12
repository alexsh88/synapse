import PageHeader from "../components/PageHeader";
import { useProjects, useTimeline } from "../lib/api";
import { nodeColor } from "../lib/nodeColors";
import ErrorBanner from "../components/ErrorBanner";

function fmt(d: string | null): string {
  if (!d) return "";
  try {
    return new Date(d).toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" });
  } catch {
    return "";
  }
}

export default function TimelinePage() {
  const { data: projects } = useProjects();
  const scopes = ["global", ...(projects?.map((p) => `project_${p.id}`) ?? [])];
  const { data: items, isLoading, isError } = useTimeline(scopes, 40);

  return (
    <div className="h-full overflow-y-auto px-8 py-7">
      <PageHeader title="Timeline" subtitle="How the knowledge accumulated — newest first." />

      {isError && <ErrorBanner className="mt-5 max-w-3xl" />}

      <ol className="relative mt-7 max-w-3xl border-l border-line/70 pl-6">
        {items?.map((it) => (
          <li key={it.id} className="relative pb-6">
            <span
              className="absolute -left-[27px] top-1.5 h-2.5 w-2.5 rounded-full border-2 border-bg"
              style={{ background: nodeColor(it.kind), boxShadow: `0 0 8px ${nodeColor(it.kind)}` }}
            />
            <div className="flex items-center gap-2 text-xs">
              <span className="font-mono uppercase tracking-wide" style={{ color: nodeColor(it.kind) }}>
                {it.kind}
              </span>
              <span className="text-muted/50">·</span>
              <span className="font-mono text-muted">{it.scope.replace("project_", "")}</span>
              <span className="ml-auto font-mono text-muted/70">{fmt(it.created_at)}</span>
            </div>
            <p className="mt-1 text-sm leading-snug text-text">{it.name}</p>
          </li>
        ))}
        {isLoading && <li className="text-sm text-muted">Loading…</li>}
        {items && items.length === 0 && <li className="text-sm text-muted">No knowledge yet.</li>}
      </ol>
    </div>
  );
}
