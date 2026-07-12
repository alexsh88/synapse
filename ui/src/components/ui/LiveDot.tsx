import { useWs } from "../../lib/ws";
import { cn } from "../../lib/cn";

export default function LiveDot() {
  const connected = useWs((s) => s.connected);
  return (
    <span className="relative flex h-2 w-2" title={connected ? "live" : "offline"}>
      {connected && (
        <span className="absolute inline-flex h-full w-full rounded-full bg-emerald-400/60 animate-breathe" />
      )}
      <span
        className={cn(
          "relative inline-flex h-2 w-2 rounded-full",
          connected ? "bg-emerald-400 shadow-[0_0_8px_1px_#34d399]" : "bg-muted",
        )}
      />
    </span>
  );
}
