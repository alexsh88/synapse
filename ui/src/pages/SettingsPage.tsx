import PageHeader from "../components/PageHeader";
import { useProjects } from "../lib/api";

export default function SettingsPage() {
  const { data: projects } = useProjects();
  return (
    <div className="h-full overflow-y-auto px-8 py-7">
      <PageHeader title="Settings" subtitle="Connected projects and system configuration." />
      <h2 className="mt-7 font-display text-lg text-text">Connected projects</h2>
      <div className="mt-3 grid max-w-3xl gap-2 sm:grid-cols-2">
        {projects?.map((p) => (
          <div key={p.id} className="rounded-xl border border-line/70 bg-surface/40 p-4">
            <div className="font-mono text-sm text-text">{p.name}</div>
            <div className="mt-2 flex flex-wrap gap-3 font-mono text-xs">
              <span className="text-muted">{p.nodes} nodes</span>
              <span style={{ color: "#f5a623" }}>{p.decisions} dec</span>
              <span style={{ color: "#2dd4bf" }}>{p.conventions} conv</span>
              <span style={{ color: "#f87171" }}>{p.lessons} less</span>
            </div>
          </div>
        ))}
      </div>
      <h2 className="mt-8 font-display text-lg text-text">Where the controls live</h2>
      <div className="mt-3 max-w-2xl space-y-2 text-sm leading-relaxed text-muted">
        <p>
          <span className="text-text">Curation</span> — graph health, duplicate clusters and
          consolidation proposals are on the <span className="font-mono text-xs">Curate</span> page.
          They run nightly in the <span className="font-mono text-xs">beat</span> container and are
          proposal-only: nothing mutates the graph on a timer.
        </p>
        <p>
          <span className="text-text">Embedder and extraction mode</span> — set by environment
          (<span className="font-mono text-xs">EXTRACTION_MODE</span>), not from the UI. The
          embedding dimension is locked at first ingestion, so changing it is a re-embedding
          migration rather than a setting.
        </p>
        <p>
          <span className="text-text">Backup and export</span> — CLI only, deliberately: they are
          destructive-adjacent and belong next to a terminal where the output can be read.
        </p>
      </div>
    </div>
  );
}
