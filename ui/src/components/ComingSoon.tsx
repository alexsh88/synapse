import type { LucideIcon } from "lucide-react";

export default function ComingSoon({
  icon: Icon, title, phase, desc,
}: { icon: LucideIcon; title: string; phase: string; desc: string }) {
  return (
    <div className="grid h-full place-items-center px-8 text-center">
      <div className="max-w-sm">
        <div className="mx-auto grid h-14 w-14 place-items-center rounded-2xl border border-line bg-surface/60">
          <Icon size={24} strokeWidth={1.5} className="text-muted" />
        </div>
        <h1 className="mt-5 font-display text-2xl font-medium text-text">{title}</h1>
        <p className="mt-2 text-sm leading-relaxed text-muted">{desc}</p>
        <span className="mt-4 inline-block rounded-full border border-accent/30 bg-accent/10 px-3 py-1 font-mono text-xs text-accent">
          {phase}
        </span>
      </div>
    </div>
  );
}
