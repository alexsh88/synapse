/** Human label for a group_id scope: "project_acme-api" → "acme-api", "global" → "global". */
export const prettyScope = (scope: string): string =>
  scope === "global" ? "global" : scope.replace(/^project_/, "");

/** Whether a scope is a project scope (vs global / agent). */
export const isProjectScope = (scope: string): boolean => scope.startsWith("project_");
