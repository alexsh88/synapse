import type { ReactNode } from "react";
import NavRail from "./NavRail";
import TopBar from "./TopBar";
import CommandPalette from "../CommandPalette";

export default function AppShell({ children }: { children: ReactNode }) {
  return (
    <div className="atmosphere grain flex h-full w-full overflow-hidden">
      <NavRail />
      <div className="flex min-w-0 flex-1 flex-col">
        <TopBar />
        <main className="relative min-h-0 flex-1 overflow-hidden">{children}</main>
      </div>
      <CommandPalette />
    </div>
  );
}
