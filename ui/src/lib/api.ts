import { useQuery } from "@tanstack/react-query";
import type {
  ApplyResult, Brief, ConnectJob, CurationHealth, CurationSuggestions, GraphSnapshot,
  NodeDetail, PendingCapture, ProjectStatus, ProjectSummary, Recalled, TimelineItem,
} from "./types";

const BASE = "/api/v1";

async function get<T>(path: string): Promise<T> {
  const res = await fetch(BASE + path);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText} — ${path}`);
  return res.json() as Promise<T>;
}

const scopeQS = (scopes: string[]) =>
  scopes.map((s) => `scope=${encodeURIComponent(s)}`).join("&");

export const useProjects = () =>
  useQuery({ queryKey: ["projects"], queryFn: () => get<ProjectSummary[]>("/projects") });

export const useProjectsStatus = () =>
  useQuery({ queryKey: ["projects-status"], queryFn: () => get<ProjectStatus[]>("/projects") });

export async function connectProject(body: {
  id: string;
  name?: string;
  path?: string;
  description?: string;
  deep_seed: boolean;
}): Promise<ConnectJob> {
  const res = await fetch(`${BASE}/projects/connect`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const detail = await res.json().catch(() => null);
    throw new Error(detail?.detail ?? `connect failed: ${res.status}`);
  }
  return res.json() as Promise<ConnectJob>;
}

export const fetchConnectJob = (jobId: string) => get<ConnectJob>(`/projects/connect/${jobId}`);

export const useGraph = (scopes: string[], opts?: { asOf?: string; includeSuperseded?: boolean }) =>
  useQuery({
    queryKey: ["graph", scopes, opts?.asOf, opts?.includeSuperseded],
    queryFn: () =>
      get<GraphSnapshot>(
        `/graph?${scopeQS(scopes)}` +
          (opts?.asOf ? `&as_of=${encodeURIComponent(opts.asOf)}` : "") +
          (opts?.includeSuperseded ? "&include_superseded=true" : ""),
      ),
  });

export const useNode = (id: string | null) =>
  useQuery({ queryKey: ["node", id], enabled: !!id, queryFn: () => get<NodeDetail>(`/graph/node/${id}`) });

export const useTimeline = (scopes: string[], limit = 50) =>
  useQuery({ queryKey: ["timeline", scopes, limit], queryFn: () => get<TimelineItem[]>(`/timeline?${scopeQS(scopes)}&limit=${limit}`) });

export const useBrief = (projectId?: string) =>
  useQuery({ queryKey: ["brief", projectId], enabled: !!projectId, queryFn: () => get<Brief>(`/brief/${projectId}`) });

export const searchKnowledge = (q: string, scope?: string) =>
  get<Recalled[]>(`/search?q=${encodeURIComponent(q)}${scope ? `&scope=${scope}` : ""}`);

export const useSearch = (q: string) =>
  useQuery({
    queryKey: ["search", q],
    queryFn: () => searchKnowledge(q),
    enabled: q.trim().length > 0,
  });

export const useCurationHealth = () =>
  useQuery({ queryKey: ["curation-health"], queryFn: () => get<CurationHealth>("/curation/health") });

export const useCurationSuggestions = () =>
  useQuery({ queryKey: ["curation-suggestions"], queryFn: () => get<CurationSuggestions>("/curation/suggestions") });

export const useCaptures = () =>
  useQuery({ queryKey: ["captures"], queryFn: () => get<PendingCapture[]>("/captures") });

export async function reviewCapture(uuid: string, action: "approve" | "dismiss") {
  const res = await fetch(`${BASE}/captures/${uuid}/${action}`, { method: "POST" });
  if (!res.ok) throw new Error(`${action} failed: ${res.status}`);
  return res.json();
}

export type ApplyAction = "merge" | "archive" | "restore";

export async function applyCuration(body: {
  action: ApplyAction;
  edge_uuid: string;
  canonical_uuid?: string;
}): Promise<ApplyResult> {
  const res = await fetch(`${BASE}/curation/apply`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`apply failed: ${res.status}`);
  return res.json() as Promise<ApplyResult>;
}

export async function remember(body: { content: string; type?: string; scope?: string }) {
  const res = await fetch(`${BASE}/knowledge`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return res.json();
}
