import { motion } from "framer-motion";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { ArrowUpRight, GitBranch, History, Activity, Copy, Archive, GitMerge, AlertTriangle, Check, X } from "lucide-react";

import PageHeader from "../components/PageHeader";
import { applyCuration, reviewCapture, useCaptures, useCurationHealth, useCurationSuggestions } from "../lib/api";
import { nodeColor } from "../lib/nodeColors";
import { prettyScope } from "../lib/scope";
import type { CurationHealth } from "../lib/types";

export default function CuratePage() {
  const { data, isLoading } = useCurationHealth();
  const { data: sugg } = useCurationSuggestions();
  const { data: captures } = useCaptures();
  const qc = useQueryClient();
  const apply = useMutation({
    mutationFn: applyCuration,
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["curation-suggestions"] });
      qc.invalidateQueries({ queryKey: ["curation-health"] });
    },
  });
  const review = useMutation({
    mutationFn: ({ uuid, action }: { uuid: string; action: "approve" | "dismiss" }) =>
      reviewCapture(uuid, action),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["captures"] });
      qc.invalidateQueries({ queryKey: ["curation-health"] });
    },
  });
  const busy = apply.isPending;

  return (
    <div className="h-full overflow-y-auto px-8 py-7">
      <PageHeader
        title="Curate"
        subtitle="Keep the brain healthy — its shape, what spans projects, and what time has superseded."
      />

      {isLoading && <p className="mt-6 text-sm text-muted">Reading the graph…</p>}

      {data && (
        <motion.div
          className="mt-6 grid max-w-5xl gap-5"
          initial="hidden"
          animate="show"
          variants={{ show: { transition: { staggerChildren: 0.06 } } }}
        >
          <Section>
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
              <Stat label="nodes" value={data.total_nodes} icon={Activity} accent />
              <Stat label="active links" value={data.active_edges} icon={GitBranch} />
              <Stat label="superseded" value={data.superseded_edges} icon={History} />
              <Stat label="cross-project" value={data.cross_project_links} icon={ArrowUpRight} />
            </div>
          </Section>

          {captures && captures.length > 0 && (
            <Section>
              <CardTitle hint="auto-captured from sessions · review to keep">
                Pending captures
              </CardTitle>
              <ul className="space-y-2.5">
                {captures.map((c) => (
                  <li key={c.uuid} className="flex items-start gap-3 rounded-lg border border-line/60 bg-surface/30 p-3">
                    <span
                      className="mt-1.5 h-2 w-2 shrink-0 rounded-full"
                      style={{ background: nodeColor(c.type), boxShadow: `0 0 7px 1px ${nodeColor(c.type)}` }}
                    />
                    <div className="min-w-0 flex-1">
                      <p className="font-display text-sm leading-snug text-text">{c.content}</p>
                      <p className="mt-1 flex flex-wrap items-center gap-2 font-mono text-[10px] text-muted">
                        <span className="capitalize">{c.type}</span>
                        <span className="text-accent/70">{c.confidence.toFixed(2)}</span>
                        <span className="rounded bg-surface-2 px-1.5 py-0.5">{prettyScope(`project_${c.project_id}`)}</span>
                        {c.reason && <span className="text-muted/60">· {c.reason}</span>}
                      </p>
                    </div>
                    <div className="flex shrink-0 gap-1.5">
                      <button
                        disabled={review.isPending}
                        onClick={() => review.mutate({ uuid: c.uuid, action: "approve" })}
                        title="keep this lesson"
                        className="flex items-center gap-1 rounded-md border border-line bg-surface-2/60 px-2 py-1 font-mono text-[10px] text-muted transition-colors hover:border-pattern/50 hover:text-pattern disabled:opacity-40"
                      >
                        <Check size={11} /> keep
                      </button>
                      <button
                        disabled={review.isPending}
                        onClick={() => review.mutate({ uuid: c.uuid, action: "dismiss" })}
                        title="dismiss"
                        className="flex items-center gap-1 rounded-md border border-line bg-surface-2/60 px-2 py-1 font-mono text-[10px] text-muted transition-colors hover:border-lesson/50 hover:text-lesson disabled:opacity-40"
                      >
                        <X size={11} /> drop
                      </button>
                    </div>
                  </li>
                ))}
              </ul>
            </Section>
          )}

          <div className="grid gap-5 lg:grid-cols-2">
            <Section>
              <CardTitle>Knowledge by type</CardTitle>
              <TypeDistribution data={data} />
            </Section>

            <Section>
              <CardTitle hint="appears in ≥2 projects">Promotion candidates</CardTitle>
              {data.promotion_candidates.length === 0 ? (
                <Empty>Nothing recurs across projects yet.</Empty>
              ) : (
                <ul className="space-y-2.5">
                  {data.promotion_candidates.map((c) => (
                    <li key={c.name} className="flex items-start gap-2.5">
                      <span
                        className="mt-1.5 h-2 w-2 shrink-0 rounded-full"
                        style={{ background: nodeColor(c.type), boxShadow: `0 0 7px 1px ${nodeColor(c.type)}` }}
                      />
                      <div className="min-w-0">
                        <p className="truncate font-display text-sm text-text">{c.name}</p>
                        <p className="mt-0.5 flex flex-wrap items-center gap-1.5 font-mono text-[10px] text-muted">
                          {c.projects.map((p) => (
                            <span key={p} className="rounded bg-surface-2 px-1.5 py-0.5">{p}</span>
                          ))}
                          <span className="text-accent/70">→ promote to global</span>
                        </p>
                      </div>
                    </li>
                  ))}
                </ul>
              )}
            </Section>
          </div>

          <Section>
            <CardTitle hint="the temporal model, made visible">Recently superseded</CardTitle>
            {data.recently_superseded.length === 0 ? (
              <Empty>No knowledge has been superseded yet — the graph is all current.</Empty>
            ) : (
              <ul className="space-y-2">
                {data.recently_superseded.map((s, i) => (
                  <li
                    key={i}
                    className="flex items-start gap-3 rounded-lg border border-line/60 bg-surface/30 px-3 py-2"
                  >
                    <History size={14} className="mt-0.5 shrink-0 text-muted/60" />
                    <div className="min-w-0 flex-1">
                      <p className="font-display text-sm leading-snug text-muted line-through decoration-line decoration-1">
                        {s.fact}
                      </p>
                      <p className="mt-0.5 font-mono text-[10px] text-muted/70">
                        {prettyScope(s.scope)}
                        {s.invalid_at && ` · superseded ${s.invalid_at.slice(0, 10)}`}
                      </p>
                    </div>
                  </li>
                ))}
              </ul>
            )}
          </Section>

          <Section>
            <CardTitle hint="cosine ≥ 0.9 · same scope">Duplicate clusters</CardTitle>
            {!sugg ? (
              <Empty>Scanning…</Empty>
            ) : sugg.duplicates.length === 0 ? (
              <Empty>No duplicates — write-time dedup is keeping the graph clean.</Empty>
            ) : (
              <ul className="space-y-3.5">
                {sugg.duplicates.map((c) => (
                  <li key={c.canonical.uuid} className="rounded-lg border border-line/60 bg-surface/30 p-3.5">
                    <div className="mb-2 flex items-center gap-2 font-mono text-[10px] text-muted">
                      <Copy size={12} className="text-accent" />
                      <span>{prettyScope(c.scope)}</span>
                      <span className="text-accent">{c.max_similarity.toFixed(3)}</span>
                      <span className="text-muted/60">keep ↓ · supersede the rest</span>
                    </div>
                    <p className="font-display text-sm leading-snug text-text">{c.canonical.fact}</p>
                    <ul className="mt-2 space-y-1.5">
                      {c.duplicates.map((d) => (
                        <li key={d.uuid} className="flex items-start gap-2.5">
                          <p className="min-w-0 flex-1 font-display text-sm leading-snug text-muted">{d.fact}</p>
                          <button
                            disabled={busy}
                            onClick={() => apply.mutate({ action: "merge", edge_uuid: d.uuid, canonical_uuid: c.canonical.uuid })}
                            className="flex shrink-0 items-center gap-1 rounded-md border border-line bg-surface-2/60 px-2 py-1 font-mono text-[10px] text-muted transition-colors hover:border-accent/40 hover:text-accent disabled:opacity-40"
                          >
                            <GitMerge size={11} /> merge
                          </button>
                        </li>
                      ))}
                    </ul>
                  </li>
                ))}
              </ul>
            )}
          </Section>

          <div className="grid gap-5 lg:grid-cols-2">
            <Section>
              <CardTitle hint="0.75–0.9 · a human glance">Needs review</CardTitle>
              {!sugg || sugg.review_pairs.length === 0 ? (
                <Empty>No ambiguous pairs to review.</Empty>
              ) : (
                <ul className="space-y-3">
                  {sugg.review_pairs.map((p, i) => (
                    <li key={i} className="rounded-lg border border-line/60 bg-surface/30 p-3">
                      <div className="mb-1.5 flex items-center gap-2 font-mono text-[10px] text-muted">
                        <AlertTriangle size={11} className="text-decision" />
                        <span>{prettyScope(p.scope)}</span>
                        <span className="text-decision">{p.similarity.toFixed(3)}</span>
                      </div>
                      <p className="font-display text-sm leading-snug text-text">{p.a.fact}</p>
                      <p className="mt-1 font-display text-sm leading-snug text-muted">{p.b.fact}</p>
                    </li>
                  ))}
                </ul>
              )}
            </Section>

            <Section>
              <CardTitle hint="active · older than 180d">Stale — archive candidates</CardTitle>
              {!sugg || sugg.stale.length === 0 ? (
                <Empty>Nothing stale — the brain is young.</Empty>
              ) : (
                <ul className="space-y-2">
                  {sugg.stale.map((s) => (
                    <li key={s.uuid} className="flex items-start gap-2.5 rounded-lg border border-line/60 bg-surface/30 px-3 py-2">
                      <div className="min-w-0 flex-1">
                        <p className="font-display text-sm leading-snug text-text">{s.fact}</p>
                        <p className="mt-0.5 font-mono text-[10px] text-muted/70">
                          {prettyScope(s.scope)}
                          {s.age_days != null && ` · ${s.age_days}d old`}
                        </p>
                      </div>
                      <button
                        disabled={busy}
                        onClick={() => apply.mutate({ action: "archive", edge_uuid: s.uuid })}
                        className="flex shrink-0 items-center gap-1 rounded-md border border-line bg-surface-2/60 px-2 py-1 font-mono text-[10px] text-muted transition-colors hover:border-accent/40 hover:text-accent disabled:opacity-40"
                      >
                        <Archive size={11} /> archive
                      </button>
                    </li>
                  ))}
                </ul>
              )}
            </Section>
          </div>

          <p className="font-mono text-[11px] leading-relaxed text-muted/60">
            Live signals. Every action is <span className="text-muted">reversible</span> and
            <span className="text-muted"> backup-first</span> — merge/archive supersede or flag (never delete),
            and a nightly Celery scan refreshes these without ever mutating on a timer.
          </p>
        </motion.div>
      )}
    </div>
  );
}

const sectionVariants = {
  hidden: { opacity: 0, y: 10 },
  show: { opacity: 1, y: 0, transition: { duration: 0.3, ease: [0.22, 1, 0.36, 1] as const } },
};

function Section({ children }: { children: React.ReactNode }) {
  return (
    <motion.section
      variants={sectionVariants}
      className="rounded-2xl border border-line/70 bg-surface/40 p-5 backdrop-blur-sm"
    >
      {children}
    </motion.section>
  );
}

function CardTitle({ children, hint }: { children: React.ReactNode; hint?: string }) {
  return (
    <h2 className="mb-4 flex items-baseline gap-2">
      <span className="font-display text-[15px] text-text">{children}</span>
      {hint && <span className="font-mono text-[10px] text-muted/60">{hint}</span>}
    </h2>
  );
}

function Stat({
  label,
  value,
  icon: Icon,
  accent,
}: {
  label: string;
  value: number;
  icon: typeof Activity;
  accent?: boolean;
}) {
  return (
    <div className="rounded-xl border border-line/60 bg-surface-2/40 p-3.5">
      <Icon size={15} className={accent ? "text-accent" : "text-muted/70"} strokeWidth={1.75} />
      <p className={"mt-2 font-mono text-2xl " + (accent ? "text-accent" : "text-text")}>
        {value.toLocaleString()}
      </p>
      <p className="font-mono text-[10px] uppercase tracking-wider text-muted/70">{label}</p>
    </div>
  );
}

function TypeDistribution({ data }: { data: CurationHealth }) {
  const max = Math.max(1, ...data.by_type.map((t) => t.count));
  if (data.by_type.length === 0) return <Empty>No typed knowledge yet.</Empty>;
  return (
    <div className="space-y-2.5">
      {data.by_type.map((t) => (
        <div key={t.type} className="flex items-center gap-3">
          <span className="w-20 shrink-0 font-mono text-[11px] capitalize text-muted">{t.type}</span>
          <span className="h-2 flex-1 overflow-hidden rounded-full bg-surface-2">
            <motion.span
              className="block h-full rounded-full"
              style={{ background: nodeColor(t.type) }}
              initial={{ width: 0 }}
              animate={{ width: `${(t.count / max) * 100}%` }}
              transition={{ duration: 0.6, ease: "easeOut" }}
            />
          </span>
          <span className="w-7 shrink-0 text-right font-mono text-[11px] text-text">{t.count}</span>
        </div>
      ))}
    </div>
  );
}

function Empty({ children }: { children: React.ReactNode }) {
  return <p className="py-2 text-sm text-muted/70">{children}</p>;
}
