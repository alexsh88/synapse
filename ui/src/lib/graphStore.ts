import { create } from "zustand";
import { KNOWLEDGE_TYPES } from "./nodeColors";

const ALL_TYPES = [...KNOWLEDGE_TYPES, "project", "entity"];

interface GraphState {
  scopeMode: string;              // "all" or a specific group_id
  enabledTypes: Set<string>;
  mode: "2d" | "3d";
  asOf: string | null;            // ISO timestamp, or null = now
  includeSuperseded: boolean;
  selectedNodeId: string | null;

  setScopeMode: (s: string) => void;
  toggleType: (t: string) => void;
  setMode: (m: "2d" | "3d") => void;
  setAsOf: (iso: string | null) => void;
  setIncludeSuperseded: (v: boolean) => void;
  select: (id: string | null) => void;
}

export const useGraphStore = create<GraphState>((set) => ({
  scopeMode: "all",
  enabledTypes: new Set(ALL_TYPES),
  mode: "2d",
  asOf: null,
  includeSuperseded: false,
  selectedNodeId: null,

  setScopeMode: (s) => set({ scopeMode: s }),
  toggleType: (t) =>
    set((st) => {
      const next = new Set(st.enabledTypes);
      next.has(t) ? next.delete(t) : next.add(t);
      return { enabledTypes: next };
    }),
  setMode: (m) => set({ mode: m }),
  setAsOf: (iso) => set({ asOf: iso }),
  setIncludeSuperseded: (v) => set({ includeSuperseded: v }),
  select: (id) => set({ selectedNodeId: id }),
}));
