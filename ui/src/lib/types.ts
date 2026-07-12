export type NodeType =
  | "decision" | "convention" | "lesson" | "research"
  | "pattern" | "tool" | "project" | "entity";

export interface GraphNode {
  id: string;
  name: string;
  type: string;
  scope: string;
  degree: number;
  summary?: string | null;
}

export interface GraphLink {
  source: string;
  target: string;
  name?: string | null;
  fact?: string | null;
}

export interface GraphSnapshot {
  nodes: GraphNode[];
  links: GraphLink[];
}

export interface NodeDetail {
  node: GraphNode;
  attributes: Record<string, unknown>;
  edges_out: GraphLink[];
  edges_in: GraphLink[];
}

export interface ProjectSummary {
  id: string;
  name: string;
  nodes: number;
  decisions: number;
  conventions: number;
  lessons: number;
}

export interface TimelineItem {
  id: string;
  kind: string;
  name: string;
  scope: string;
  created_at: string | null;
}

export interface Recalled {
  fact: string;
  score: number;
  scope: string;
  uuid: string;
  valid_at?: string | null;
  /** per-strategy score breakdown: relevance / recency / confidence / connectivity */
  components?: Record<string, number>;
}

export interface TypeCount {
  type: string;
  count: number;
}

export interface PromotionCandidate {
  name: string;
  type: string;
  projects: string[];
}

export interface SupersededItem {
  fact: string;
  scope: string;
  invalid_at?: string | null;
}

export interface FactRef {
  uuid: string;
  fact: string;
}

export interface DuplicateCluster {
  scope: string;
  canonical: FactRef;
  duplicates: FactRef[];
  max_similarity: number;
}

export interface StaleItem {
  uuid: string;
  fact: string;
  scope: string;
  created_at?: string | null;
  age_days?: number | null;
}

export interface ReviewPair {
  scope: string;
  a: FactRef;
  b: FactRef;
  similarity: number;
}

export interface CurationSuggestions {
  duplicates: DuplicateCluster[];
  stale: StaleItem[];
  review_pairs: ReviewPair[];
  generated_at?: string | null;
}

export interface ApplyResult {
  ok: boolean;
  action: string;
  edge_uuid: string;
  backup_path?: string | null;
  detail: string;
}

export interface PendingCapture {
  uuid: string;
  project_id: string;
  content: string;
  type: string;
  confidence: number;
  reason: string;
}

export interface CurationHealth {
  total_nodes: number;
  active_edges: number;
  superseded_edges: number;
  cross_project_links: number;
  by_type: TypeCount[];
  promotion_candidates: PromotionCandidate[];
  recently_superseded: SupersededItem[];
}

export interface ProjectStatus {
  id: string;
  name: string;
  cluster: string;
  connected: boolean;
  hook: boolean;
  exists: boolean;
  nodes: number;
  decisions: number;
  conventions: number;
  lessons: number;
}

export interface ConnectJob {
  job_id: string;
  project: string;
  state: "running" | "done" | "error";
  done: number;
  total: number;
  stored: number;
  actions?: string[];
  entity?: string;
  error?: string;
}

export interface Brief {
  project_id: string;
  project_summary: string;
  active_conventions: string[];
  key_decisions: string[];
  relevant_lessons: string[];
  cross_project_knowledge: string[];
}
