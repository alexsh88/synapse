import { type FormEvent, useEffect, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { Boxes, CornerDownLeft, Loader2, Check, X, AlertTriangle } from "lucide-react";

import { connectProject, fetchConnectJob } from "../../lib/api";
import type { ConnectJob, ProjectStatus } from "../../lib/types";

type Phase = "form" | "connecting" | "done" | "error";

export default function ConnectProjectModal({
  open,
  onClose,
  prefill,
  onConnected,
}: {
  open: boolean;
  onClose: () => void;
  prefill?: ProjectStatus | null;
  onConnected: () => void;
}) {
  const [id, setId] = useState("");
  const [name, setName] = useState("");
  const [path, setPath] = useState("");
  const [deepSeed, setDeepSeed] = useState(true);
  const [phase, setPhase] = useState<Phase>("form");
  const [job, setJob] = useState<ConnectJob | null>(null);
  const [error, setError] = useState("");
  const cancelled = useRef(false);

  // Reset on open; prefill from a known project when reconnecting/seeding.
  useEffect(() => {
    if (open) {
      cancelled.current = false;
      setId(prefill?.id ?? "");
      setName(prefill?.name ?? "");
      setPath("");
      setDeepSeed(true);
      setPhase("form");
      setJob(null);
      setError("");
    } else {
      cancelled.current = true;
    }
  }, [open, prefill]);

  async function submit(e: FormEvent) {
    e.preventDefault();
    if (!id.trim()) return;
    setPhase("connecting");
    setError("");
    try {
      let j = await connectProject({
        id: id.trim(),
        name: name.trim() || undefined,
        path: path.trim() || undefined,
        deep_seed: deepSeed,
      });
      setJob(j);
      // Poll until the background deep-seed finishes.
      while (j.state === "running" && !cancelled.current) {
        await new Promise((r) => setTimeout(r, 1200));
        if (cancelled.current) return;
        j = await fetchConnectJob(j.job_id);
        setJob(j);
      }
      setPhase(j.state === "error" ? "error" : "done");
      if (j.state === "error") setError(j.error ?? "deep-seed failed");
      onConnected();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setPhase("error");
    }
  }

  const pct = job && job.total > 0 ? Math.round((job.done / job.total) * 100) : 0;

  return (
    <AnimatePresence>
      {open && (
        <motion.div
          className="fixed inset-0 z-50 flex items-start justify-center px-4 pt-[14vh]"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.18 }}
          onClick={onClose}
        >
          <div className="absolute inset-0 bg-bg/70 backdrop-blur-sm" />
          <motion.div
            className="relative w-full max-w-lg overflow-hidden rounded-2xl border border-line bg-surface/95 shadow-[0_24px_80px_-12px_rgba(0,0,0,0.8),0_0_0_1px_rgba(34,211,238,0.06)]"
            initial={{ opacity: 0, y: -10, scale: 0.985 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -8, scale: 0.99 }}
            transition={{ duration: 0.22, ease: [0.22, 1, 0.36, 1] }}
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center gap-2.5 border-b border-line/70 px-5 py-3.5">
              <Boxes size={16} className="text-accent" strokeWidth={1.75} />
              <span className="font-display text-[15px] text-text">Connect a project</span>
              <button onClick={onClose} className="ml-auto text-muted hover:text-text">
                <X size={16} />
              </button>
            </div>

            {phase === "form" && (
              <form onSubmit={submit} className="space-y-3.5 p-5">
                <Field label="Project id" hint="e.g. acme-jobs (the SYNAPSE_PROJECT_ID)">
                  <input
                    value={id}
                    onChange={(e) => setId(e.target.value)}
                    readOnly={!!prefill}
                    autoFocus={!prefill}
                    placeholder="my-project"
                    className="w-full rounded-lg border border-line bg-surface-2/60 px-3 py-2 text-sm text-text placeholder:text-muted/50 focus:border-accent/40 focus:outline-none read-only:opacity-60"
                  />
                </Field>
                <Field label="Name">
                  <input
                    value={name}
                    onChange={(e) => setName(e.target.value)}
                    placeholder={id || "My Project"}
                    className="w-full rounded-lg border border-line bg-surface-2/60 px-3 py-2 text-sm text-text placeholder:text-muted/50 focus:border-accent/40 focus:outline-none"
                  />
                </Field>
                {!prefill && (
                  <Field label="Path" hint="absolute path under the projects root">
                    <input
                      value={path}
                      onChange={(e) => setPath(e.target.value)}
                      placeholder="C:/Users/dev/dev/projects/my-project"
                      className="w-full rounded-lg border border-line bg-surface-2/60 px-3 py-2 font-mono text-xs text-text placeholder:text-muted/50 focus:border-accent/40 focus:outline-none"
                    />
                  </Field>
                )}
                <label className="flex cursor-pointer items-center gap-2.5 pt-1 text-sm text-muted">
                  <input
                    type="checkbox"
                    checked={deepSeed}
                    onChange={(e) => setDeepSeed(e.target.checked)}
                    className="accent-accent"
                  />
                  Deep-seed — read the project's docs and extract knowledge now
                </label>
                <button
                  type="submit"
                  className="mt-1 flex w-full items-center justify-center gap-2 rounded-lg bg-accent/15 px-3 py-2.5 text-sm font-medium text-accent transition-colors hover:bg-accent/25"
                >
                  Connect <CornerDownLeft size={13} />
                </button>
              </form>
            )}

            {phase === "connecting" && (
              <div className="space-y-4 p-6">
                <div className="flex items-center gap-2.5 text-sm text-text">
                  <Loader2 size={16} className="animate-spin text-accent" />
                  {job && job.total > 0 ? "Extracting knowledge from docs…" : "Wiring project…"}
                </div>
                <div className="h-1.5 w-full overflow-hidden rounded-full bg-surface-2">
                  <motion.div
                    className="h-full rounded-full bg-accent"
                    animate={{ width: `${job && job.total ? pct : 12}%` }}
                    transition={{ duration: 0.4 }}
                  />
                </div>
                <p className="font-mono text-[11px] text-muted">
                  {job?.total
                    ? `${job.done}/${job.total} chunks · ${job.stored} facts stored`
                    : "writing .mcp.json · brief hook · CLAUDE.md · Project entity"}
                </p>
              </div>
            )}

            {phase === "done" && (
              <div className="space-y-3 p-6">
                <div className="flex items-center gap-2.5 text-sm text-text">
                  <Check size={16} className="text-pattern" /> Connected.
                </div>
                <ul className="space-y-1 font-mono text-[11px] text-muted">
                  {(job?.actions ?? []).map((a, i) => <li key={i}>· {a}</li>)}
                  {job?.entity && <li>· Project entity: {job.entity}</li>}
                  {!!job?.stored && <li>· {job.stored} facts extracted from docs</li>}
                </ul>
                <button onClick={onClose} className="rounded-lg border border-line px-3 py-1.5 text-sm text-muted hover:text-text">
                  Done
                </button>
              </div>
            )}

            {phase === "error" && (
              <div className="space-y-3 p-6">
                <div className="flex items-center gap-2.5 text-sm text-lesson">
                  <AlertTriangle size={16} /> Couldn't connect
                </div>
                <p className="font-mono text-[11px] text-muted">{error}</p>
                <button onClick={() => setPhase("form")} className="rounded-lg border border-line px-3 py-1.5 text-sm text-muted hover:text-text">
                  Back
                </button>
              </div>
            )}
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}

function Field({ label, hint, children }: { label: string; hint?: string; children: React.ReactNode }) {
  return (
    <label className="block">
      <span className="mb-1 flex items-baseline gap-2 text-xs text-muted">
        {label}
        {hint && <span className="text-[10px] text-muted/60">{hint}</span>}
      </span>
      {children}
    </label>
  );
}
