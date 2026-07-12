import { useEffect, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { useQueryClient } from "@tanstack/react-query";
import { motion } from "framer-motion";
import { Plus, Link2, Zap } from "lucide-react";

import PageHeader from "../components/PageHeader";
import ConnectProjectModal from "../components/domain/ConnectProjectModal";
import { useProjectsStatus } from "../lib/api";
import { nodeColor } from "../lib/nodeColors";
import type { ProjectStatus } from "../lib/types";

export default function ProjectsPage() {
  const { data } = useProjectsStatus();
  const qc = useQueryClient();
  const [params, setParams] = useSearchParams();
  const [modalOpen, setModalOpen] = useState(false);
  const [prefill, setPrefill] = useState<ProjectStatus | null>(null);

  // ⌘K "Connect a project…" deep-links here with ?connect=1.
  useEffect(() => {
    if (params.get("connect") === "1") {
      setPrefill(null);
      setModalOpen(true);
      params.delete("connect");
      setParams(params, { replace: true });
    }
  }, [params, setParams]);

  const onConnected = () => {
    qc.invalidateQueries({ queryKey: ["projects-status"] });
    qc.invalidateQueries({ queryKey: ["projects"] });
  };

  const open = (p: ProjectStatus | null) => {
    setPrefill(p);
    setModalOpen(true);
  };

  const connected = (data ?? []).filter((p) => p.connected).length;

  return (
    <div className="h-full overflow-y-auto px-8 py-7">
      <div className="flex items-start justify-between">
        <PageHeader
          title="Projects"
          subtitle={`${connected} of ${data?.length ?? 0} connected to the brain.`}
        />
        <button
          onClick={() => open(null)}
          className="flex items-center gap-2 rounded-lg bg-accent/15 px-3.5 py-2 text-sm font-medium text-accent transition-colors hover:bg-accent/25"
        >
          <Plus size={15} /> Connect project
        </button>
      </div>

      <motion.div
        className="mt-6 grid max-w-5xl grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3"
        initial="hidden"
        animate="show"
        variants={{ show: { transition: { staggerChildren: 0.04 } } }}
      >
        {(data ?? []).map((p) => (
          <motion.button
            key={p.id}
            onClick={() => open(p)}
            variants={{ hidden: { opacity: 0, y: 8 }, show: { opacity: 1, y: 0 } }}
            className="group rounded-xl border border-line/70 bg-surface/40 p-4 text-left transition-colors hover:border-accent/40 hover:bg-surface-2/40"
          >
            <div className="flex items-center gap-2">
              <span
                className="h-2 w-2 shrink-0 rounded-full"
                style={
                  p.connected
                    ? { background: "var(--color-accent)", boxShadow: "0 0 7px 1px var(--color-accent)" }
                    : { background: "var(--color-muted)", opacity: 0.4 }
                }
              />
              <span className="truncate font-display text-[15px] text-text">{p.name}</span>
              <span className="ml-auto font-mono text-[10px] uppercase tracking-wider text-muted/60">
                {p.cluster}
              </span>
            </div>

            <div className="mt-3 flex items-center gap-3 font-mono text-[10px] text-muted">
              <span className="flex items-center gap-1">
                <Link2 size={11} className={p.connected ? "text-accent" : "text-muted/40"} />
                {p.connected ? "connected" : "not connected"}
              </span>
              {p.hook && (
                <span className="flex items-center gap-1 text-pattern">
                  <Zap size={11} /> brief hook
                </span>
              )}
            </div>

            <div className="mt-3 flex items-center gap-3">
              <span className="font-mono text-xl text-text">{p.nodes}</span>
              <span className="font-mono text-[10px] text-muted/70">nodes</span>
              <span className="ml-auto flex gap-2 font-mono text-[10px] text-muted">
                <Count n={p.decisions} type="decision" label="dec" />
                <Count n={p.conventions} type="convention" label="conv" />
                <Count n={p.lessons} type="lesson" label="les" />
              </span>
            </div>

            <p className="mt-2 font-mono text-[10px] text-muted/0 transition-colors group-hover:text-accent/70">
              {p.connected ? "click to re-seed →" : "click to connect →"}
            </p>
          </motion.button>
        ))}
      </motion.div>

      <ConnectProjectModal
        open={modalOpen}
        onClose={() => setModalOpen(false)}
        prefill={prefill}
        onConnected={onConnected}
      />
    </div>
  );
}

function Count({ n, type, label }: { n: number; type: string; label: string }) {
  return (
    <span className="flex items-center gap-1">
      <span className="h-1.5 w-1.5 rounded-full" style={{ background: nodeColor(type) }} />
      {n} {label}
    </span>
  );
}
