export const NODE_COLORS: Record<string, string> = {
  decision: "#f5a623",
  convention: "#2dd4bf",
  lesson: "#f87171",
  research: "#a78bfa",
  pattern: "#4ade80",
  tool: "#60a5fa",
  project: "#22d3ee",
  entity: "#8b919a",
};

export const nodeColor = (type: string): string => NODE_COLORS[type] ?? NODE_COLORS.entity;

export const KNOWLEDGE_TYPES = [
  "decision", "convention", "lesson", "research", "pattern", "tool",
] as const;
