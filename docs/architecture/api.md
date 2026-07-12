# API (Phase 5)

FastAPI read/write + WebSocket over the `KnowledgeEngine`. Routes are thin; logic
lives in `synapse/core` (engine + `GraphService`). Base path `/api/v1`. CORS allows
the Vite dev origin (`http://localhost:5173`).

## REST endpoints

| Method | Path | Purpose | Returns |
|---|---|---|---|
| GET | `/health` | liveness | `{"status":"ok"}` |
| GET | `/graph?scope=&types=&as_of=` | force-graph data for a scope set | `GraphSnapshot` |
| GET | `/graph/node/{id}` | node + connected facts | `NodeDetail` |
| GET | `/projects` | connected projects + counts | `[ProjectSummary]` |
| GET | `/timeline?scope=&type=&limit=` | reverse-chron knowledge events | `[TimelineItem]` |
| GET | `/search?q=&scope=&limit=` | ranked search across all scopes | `[Recalled]` |
| GET | `/recall?q=&project=&as_of=&limit=` | scoped recall (global+project) | `[Recalled]` |
| GET | `/brief/{project_id}` | session-start briefing | `Brief` |
| POST | `/knowledge` | remember `{content,type?,scope?}` | `WriteResult` |
| PATCH | `/knowledge/{id}` | update `{content}` (supersede) | `{...}` |
| DELETE | `/knowledge/{id}?reason=` | forget (temporal end) | `{...}` |
| GET | `/curation/health` | knowledge-health metrics (Phase 9/10) | `HealthReport` |

### Response shapes (Pydantic)

```
GraphSnapshot { nodes: [GraphNode], links: [GraphLink] }
GraphNode  { id, name, type, scope, degree, summary? }     # type = lowercased label
GraphLink  { source, target, name, fact }                  # source/target = node ids
NodeDetail { node: GraphNode, attributes: {}, edges_out: [GraphLink], edges_in: [GraphLink] }
ProjectSummary { id, name, nodes, decisions, conventions, lessons }
TimelineItem { id, kind, name, scope, created_at, valid_at?, supersedes? }
```
`Recalled`, `Brief`, `WriteResult` reuse the core models. Query params: `scope` repeatable
(`?scope=global&scope=project_acme-api`); `types` comma-list of node types; `as_of` ISO 8601.

## WebSocket — `/ws`

Real-time "aliveness." Clients connect; the server pushes a JSON event on every successful
write. Backed by an in-process `EventBus` (asyncio pub/sub) + `ConnectionManager`.

```
→ on connect: {"type":"hello","nodes":<count>}
→ on write:   {"type":"knowledge.added"|"knowledge.updated"|"knowledge.forgotten",
               "id": <episode/edge id>, "scope": <group_id>, "summary": <short>}
```
The Graph Explorer (Phase 7) listens and animates new nodes (pulse+glow). The top bar's
live dot reflects connection state.

## Errors
422 (validation), 404 (unknown id), 500 (engine error — logged, generic body). Write routes
never leak the engine's internal exceptions; they return the engine's structured result or a
clean error. Read routes that hit an empty graph return empty lists, not errors.

## Testing
`TestClient` with `app.dependency_overrides[get_engine]` → a fake engine/`GraphService`
(no live Neo4j/Ollama/Claude in unit tests). WebSocket tested via `TestClient.websocket_connect`.
Live boot verified by `uvicorn synapse.api.main:app` + `curl /api/v1/graph?scope=global`.
