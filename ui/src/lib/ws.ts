import { create } from "zustand";

export interface KnowledgeEvent {
  type: "knowledge.added" | "knowledge.updated" | "knowledge.forgotten" | "hello";
  id?: string;
  scope?: string;
  summary?: string;
}

interface WsState {
  connected: boolean;
  events: KnowledgeEvent[];
  /** increments on each knowledge.added — components can react to "the brain grew". */
  addedCount: number;
}

export const useWs = create<WsState>(() => ({ connected: false, events: [], addedCount: 0 }));

// Map of node id → timestamp (ms) of when it was received. Used for entry-pulse animation.
const newNodeTimestamps = new Map<string, number>();

export function getNewNodeTimestamps(): ReadonlyMap<string, number> {
  return newNodeTimestamps;
}

/** Prune entries older than 2 seconds. Called internally via lazy interval. */
function pruneNewNodeTimestamps(): void {
  const now = Date.now();
  for (const [id, ts] of newNodeTimestamps) {
    if (now - ts > 2000) newNodeTimestamps.delete(id);
  }
}

/** Initialize lazy pruning interval (runs every 1000ms). */
function initPruningInterval(): void {
  if (pruningIntervalId !== null) return;  // already running
  pruningIntervalId = window.setInterval(() => {
    pruneNewNodeTimestamps();
  }, 1000);
}

let pruningIntervalId: number | null = null;

let socket: WebSocket | null = null;
let retries = 0;

export function connectWs(): void {
  if (socket && socket.readyState <= WebSocket.OPEN) return;
  const proto = location.protocol === "https:" ? "wss" : "ws";
  socket = new WebSocket(`${proto}://${location.host}/ws`);

  socket.onopen = () => {
    retries = 0;
    initPruningInterval();  // start lazy pruning on first connection
    useWs.setState({ connected: true });
  };
  socket.onclose = () => {
    useWs.setState({ connected: false });
    const delay = Math.min(2000 * Math.pow(2, retries), 30000);
    retries += 1;
    setTimeout(connectWs, delay);
  };
  socket.onmessage = (e) => {
    const msg = JSON.parse(e.data as string) as KnowledgeEvent;
    if (msg.type === "hello") return;
    if (msg.type === "knowledge.added" && msg.id) {
      newNodeTimestamps.set(msg.id, Date.now());
    }
    useWs.setState((s) => ({
      events: [msg, ...s.events].slice(0, 50),
      addedCount: msg.type === "knowledge.added" ? s.addedCount + 1 : s.addedCount,
    }));
  };
}
