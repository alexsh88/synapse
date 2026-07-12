import { Suspense, lazy } from "react";

import GraphControls from "../components/graph/GraphControls";
import NodeDetailPanel from "../components/graph/NodeDetailPanel";
import Constellation from "../components/Constellation";
import ErrorBanner from "../components/ErrorBanner";
import { useGraphStore } from "../lib/graphStore";
import { useGraph, useProjects } from "../lib/api";
import { useMemo } from "react";

const GraphExplorer = lazy(() => import("../components/graph/GraphExplorer"));

function GraphErrorBoundary() {
  const { scopeMode, asOf, includeSuperseded } = useGraphStore();
  const { data: projects } = useProjects();
  const scopes = useMemo(
    () => (scopeMode === "all" ? ["global", ...(projects?.map((p) => `project_${p.id}`) ?? [])] : [scopeMode]),
    [scopeMode, projects],
  );
  const { isError } = useGraph(scopes, { asOf: asOf ?? undefined, includeSuperseded });
  if (!isError) return null;
  return (
    <div className="absolute left-1/2 top-6 z-30 -translate-x-1/2">
      <ErrorBanner />
    </div>
  );
}

export default function GraphPage() {
  return (
    <div className="relative h-full w-full">
      <Suspense
        fallback={
          <div className="absolute inset-0">
            <Constellation />
          </div>
        }
      >
        <GraphExplorer />
      </Suspense>
      <GraphErrorBoundary />
      <GraphControls />
      <NodeDetailPanel />
    </div>
  );
}
