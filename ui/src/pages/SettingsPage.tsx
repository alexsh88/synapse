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
      <p className="mt-7 max-w-2xl text-sm leading-relaxed text-muted">
        Embedder configuration, curation schedule, and backup/export controls arrive with the
        curation engine in Phases 9–10.
      </p>
    </div>
  );
}
