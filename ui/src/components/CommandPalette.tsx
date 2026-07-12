import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Command } from "cmdk";
import { AnimatePresence, motion } from "framer-motion";
import {
  Search as SearchIcon,
  Waypoints,
  FileText,
  Clock,
  Sparkles,
  Boxes,
  Settings,
  CornerDownLeft,
} from "lucide-react";

import { usePalette } from "../lib/commandStore";
import { useSearch } from "../lib/api";
import { prettyScope } from "../lib/scope";

const NAV = [
  { to: "/graph", icon: Waypoints, label: "Graph", hint: "The living graph" },
  { to: "/documents", icon: FileText, label: "Documents", hint: "Browse knowledge" },
  { to: "/timeline", icon: Clock, label: "Timeline", hint: "What changed, when" },
  { to: "/curate", icon: Sparkles, label: "Curate", hint: "Keep the brain healthy" },
  { to: "/projects", icon: Boxes, label: "Projects", hint: "Connect & manage" },
  { to: "/projects?connect=1", icon: Boxes, label: "Connect a project…", hint: "Wire + seed a new project" },
  { to: "/settings", icon: Settings, label: "Settings", hint: "Configuration" },
] as const;

export default function CommandPalette() {
  const { open, setOpen } = usePalette();
  const navigate = useNavigate();
  const [q, setQ] = useState("");

  // Global ⌘K / Ctrl+K toggle — registered once (palette mounts once in AppShell).
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        usePalette.getState().toggle();
      }
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, []);

  const { data: results = [], isFetching } = useSearch(q);

  useEffect(() => {
    if (!open) setQ("");
  }, [open]);

  const go = (to: string) => {
    navigate(to);
    setOpen(false);
  };
  const runSearch = () => go(`/search?q=${encodeURIComponent(q)}`);
  const trimmed = q.trim();

  return (
    <AnimatePresence>
      {open && (
        <motion.div
          className="fixed inset-0 z-50 flex items-start justify-center px-4 pt-[14vh]"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.18 }}
          onClick={() => setOpen(false)}
        >
          <div className="absolute inset-0 bg-bg/70 backdrop-blur-sm" />
          <motion.div
            className="relative w-full max-w-xl overflow-hidden rounded-2xl border border-line bg-surface/95 shadow-[0_24px_80px_-12px_rgba(0,0,0,0.8),0_0_0_1px_rgba(34,211,238,0.06)]"
            initial={{ opacity: 0, y: -10, scale: 0.985 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -8, scale: 0.99 }}
            transition={{ duration: 0.22, ease: [0.22, 1, 0.36, 1] }}
            onClick={(e) => e.stopPropagation()}
          >
            <Command
              shouldFilter={false}
              loop
              onKeyDown={(e) => {
                if (e.key === "Escape") setOpen(false);
              }}
            >
              <div className="flex items-center gap-3 border-b border-line/70 px-4">
                <SearchIcon size={17} className="shrink-0 text-muted" strokeWidth={1.75} />
                <Command.Input
                  value={q}
                  onValueChange={setQ}
                  autoFocus
                  placeholder="Search the brain, or jump to a view…"
                  className="h-13 w-full bg-transparent py-3.5 text-[15px] text-text placeholder:text-muted/60 focus:outline-none"
                />
                <kbd className="hidden rounded border border-line bg-bg px-1.5 py-0.5 font-mono text-[10px] text-muted sm:block">
                  esc
                </kbd>
              </div>

              <Command.List className="max-h-[52vh] overflow-y-auto p-2">
                {!trimmed && (
                  <Command.Group
                    heading="Jump to"
                    className="px-1 [&_[cmdk-group-heading]]:px-2 [&_[cmdk-group-heading]]:py-1.5 [&_[cmdk-group-heading]]:font-mono [&_[cmdk-group-heading]]:text-[10px] [&_[cmdk-group-heading]]:uppercase [&_[cmdk-group-heading]]:tracking-wider [&_[cmdk-group-heading]]:text-muted/70"
                  >
                    {NAV.map(({ to, icon: Icon, label, hint }) => (
                      <Command.Item
                        key={to}
                        value={`nav ${label}`}
                        onSelect={() => go(to)}
                        className="flex cursor-pointer items-center gap-3 rounded-lg px-2.5 py-2 text-sm text-muted transition-colors data-[selected=true]:bg-surface-2 data-[selected=true]:text-text"
                      >
                        <Icon size={16} strokeWidth={1.75} className="text-muted" />
                        <span className="text-text">{label}</span>
                        {hint && <span className="ml-auto text-xs text-muted/70">{hint}</span>}
                      </Command.Item>
                    ))}
                  </Command.Group>
                )}

                {trimmed && (
                  <>
                    <Command.Item
                      value="__search_all"
                      onSelect={runSearch}
                      className="flex cursor-pointer items-center gap-3 rounded-lg px-2.5 py-2 text-sm transition-colors data-[selected=true]:bg-surface-2"
                    >
                      <SearchIcon size={16} className="text-accent" strokeWidth={1.75} />
                      <span className="text-text">
                        Search for <span className="font-medium text-accent">"{trimmed}"</span>
                      </span>
                      <CornerDownLeft size={13} className="ml-auto text-muted/70" />
                    </Command.Item>

                    {isFetching && results.length === 0 && (
                      <div className="px-3 py-6 text-center text-sm text-muted">Searching the brain…</div>
                    )}
                    {!isFetching && results.length === 0 && (
                      <Command.Empty className="px-3 py-6 text-center text-sm text-muted">
                        No matches yet.
                      </Command.Empty>
                    )}

                    {results.length > 0 && (
                      <Command.Group
                        heading="Knowledge"
                        className="mt-1 [&_[cmdk-group-heading]]:px-2 [&_[cmdk-group-heading]]:py-1.5 [&_[cmdk-group-heading]]:font-mono [&_[cmdk-group-heading]]:text-[10px] [&_[cmdk-group-heading]]:uppercase [&_[cmdk-group-heading]]:tracking-wider [&_[cmdk-group-heading]]:text-muted/70"
                      >
                        {results.slice(0, 7).map((r) => (
                          <Command.Item
                            key={r.uuid}
                            value={r.uuid}
                            onSelect={runSearch}
                            className="flex cursor-pointer items-start gap-3 rounded-lg px-2.5 py-2 transition-colors data-[selected=true]:bg-surface-2"
                          >
                            <span className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-accent/80 shadow-[0_0_6px_1px_var(--color-accent)]" />
                            <span className="min-w-0 flex-1">
                              <span className="line-clamp-2 font-display text-sm leading-snug text-text">
                                {r.fact}
                              </span>
                              <span className="mt-0.5 flex items-center gap-2 font-mono text-[10px] text-muted">
                                <span className="text-accent">{r.score.toFixed(2)}</span>
                                <span>{prettyScope(r.scope)}</span>
                              </span>
                            </span>
                          </Command.Item>
                        ))}
                      </Command.Group>
                    )}
                  </>
                )}
              </Command.List>
            </Command>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
