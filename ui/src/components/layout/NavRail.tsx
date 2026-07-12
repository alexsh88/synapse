import { NavLink } from "react-router-dom";
import { Waypoints, FileText, Clock, Search, Sparkles, Boxes, Settings } from "lucide-react";
import { cn } from "../../lib/cn";

const ITEMS = [
  { to: "/graph", icon: Waypoints, label: "Graph" },
  { to: "/documents", icon: FileText, label: "Documents" },
  { to: "/timeline", icon: Clock, label: "Timeline" },
  { to: "/search", icon: Search, label: "Search" },
  { to: "/curate", icon: Sparkles, label: "Curate" },
  { to: "/projects", icon: Boxes, label: "Projects" },
  { to: "/settings", icon: Settings, label: "Settings" },
];

export default function NavRail() {
  return (
    <nav className="z-20 flex w-14 flex-col items-center gap-1 border-r border-line/70 bg-surface/40 py-3 backdrop-blur-sm">
      <div className="mb-3 grid h-9 w-9 place-items-center">
        <span className="h-2.5 w-2.5 rounded-full bg-accent shadow-[0_0_14px_3px_var(--color-accent)]" />
      </div>
      {ITEMS.map(({ to, icon: Icon, label }) => (
        <NavLink
          key={to}
          to={to}
          className={({ isActive }) =>
            cn(
              "group relative grid h-10 w-10 place-items-center rounded-lg text-muted transition-colors",
              "hover:bg-surface-2 hover:text-text",
              isActive && "text-accent",
            )
          }
        >
          {({ isActive }) => (
            <>
              {isActive && (
                <span className="absolute left-0 top-1/2 h-5 w-[2px] -translate-y-1/2 rounded-full bg-accent shadow-[0_0_8px_1px_var(--color-accent)]" />
              )}
              <Icon size={18} strokeWidth={1.75} />
              <span className="pointer-events-none absolute left-12 z-30 whitespace-nowrap rounded-md border border-line bg-surface-2 px-2 py-1 text-xs text-text opacity-0 shadow-lg transition-opacity group-hover:opacity-100">
                {label}
              </span>
            </>
          )}
        </NavLink>
      ))}
    </nav>
  );
}
