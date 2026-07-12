import type { ButtonHTMLAttributes } from "react";
import { cn } from "../../lib/cn";

type Variant = "default" | "ghost" | "accent";

const VARIANTS: Record<Variant, string> = {
  default: "border border-line bg-surface-2 text-text hover:border-muted/50",
  ghost: "text-muted hover:bg-surface-2 hover:text-text",
  accent: "bg-accent/15 text-accent border border-accent/30 hover:bg-accent/25",
};

export function Button({
  className,
  variant = "default",
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement> & { variant?: Variant }) {
  return (
    <button
      className={cn(
        "inline-flex items-center gap-2 rounded-lg px-3 py-1.5 text-sm font-medium transition-colors",
        "focus:outline-none focus-visible:ring-2 focus-visible:ring-accent/50 disabled:opacity-50",
        VARIANTS[variant],
        className,
      )}
      {...props}
    />
  );
}
