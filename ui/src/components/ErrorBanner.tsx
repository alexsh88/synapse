import { cn } from "../lib/cn";

interface ErrorBannerProps {
  className?: string;
  message?: string;
}

/**
 * Muted error state — palette-consistent (bg #0a0c10, surface #13161c, accent cyan #22d3ee).
 * Tone: "Couldn't reach the brain."
 */
export default function ErrorBanner({
  className,
  message = "Couldn't reach the brain. Check that Synapse is running.",
}: ErrorBannerProps) {
  return (
    <div
      className={cn(
        "flex items-center gap-3 rounded-lg border border-line bg-surface px-4 py-3 text-sm",
        className,
      )}
    >
      <span
        className="h-2 w-2 shrink-0 rounded-full"
        style={{ background: "#22d3ee", boxShadow: "0 0 8px 1px #22d3ee55" }}
      />
      <span className="text-muted">{message}</span>
    </div>
  );
}
