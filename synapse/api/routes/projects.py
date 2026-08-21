"""Projects routes — status list + one-click connect (Phase 5 / 12).

GET /projects        — every registered project with connection status + knowledge counts.
POST /projects/connect — write the wiring files (filesystem only) and return a job id; ALL graph
                         writes — the Project entity and the doc deep-seed — run in the background.
GET /projects/connect/{job_id} — poll seed progress (also streamed over the WebSocket).
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from synapse.api.deps import get_engine
from synapse.api.events import KnowledgeEvent, bus
from synapse.core import registry
from synapse.core.project_connector import ProjectConnector
from synapse.models.api import ConnectJobResponse

logger = logging.getLogger("synapse.api.projects")
router = APIRouter(tags=["projects"])

_connector = ProjectConnector()
_JOBS: dict[str, dict] = {}


def _evict_old_jobs() -> None:
    """Drop completed jobs older than 1 hour to prevent unbounded _JOBS growth."""
    import time
    now = time.monotonic()
    to_delete = [
        jid for jid, job in list(_JOBS.items())
        if job.get("state") in ("done", "error") and now - job.get("_created_at", now) > 3600
    ]
    for jid in to_delete:
        del _JOBS[jid]


class ConnectBody(BaseModel):
    id: str
    name: str | None = None
    path: str | None = None
    description: str | None = None
    deep_seed: bool = True


def _resolve_folder(host_path: str) -> Path:
    """The project's IO folder, searched across every configured root; 404 if it is under none.

    Only the folder NAME survives resolution — the container reaches host directories through bind
    mounts, so a host path has to be re-rooted at its mount point. That rewrite used to target one
    hardcoded root, which left every project outside it unconnectable and reported the failure as
    a missing folder at a CONTAINER path the caller had never typed. Traversal is still rejected,
    now per root.
    """
    name = registry.folder_name(host_path)
    searched: list[str] = []
    for root in registry.project_roots():
        root = root.resolve()
        folder = (root / name).resolve()
        if root not in folder.parents and folder != root:
            raise HTTPException(status_code=400, detail="project path escapes its projects root")
        if folder.is_dir():
            return folder
        searched.append(str(root))
    raise HTTPException(
        status_code=404,
        detail=f"no folder named {name!r} under any configured project root — searched: "
               f"{', '.join(searched)}",
    )


@router.get("/projects")
async def projects(engine=Depends(get_engine)):
    counts = {p.id: p for p in await engine.graph.projects()}
    meta_all = registry.all_projects()
    # Union: registry (+ UI-added overlay) AND any project that has knowledge in the graph
    # (so a freshly-connected project shows up immediately, even before it's in the overlay).
    ids = list(meta_all) + [pid for pid in counts if pid not in meta_all]
    out = []
    for pid in ids:
        meta = meta_all.get(pid)
        if meta:
            folder = registry.project_folder(meta["path"])
            name, cluster = meta["name"], meta["cluster"]
        else:
            folder = registry.project_folder(pid)                # graph-only: assume folder == id
            name, cluster = pid.replace("-", " ").title(), "added"
        st = _connector.status(folder)
        c = counts.get(pid)
        out.append({
            "id": pid, "name": name, "cluster": cluster,
            "connected": st["connected"], "hook": st["hook"], "exists": st["exists"],
            "nodes": c.nodes if c else 0, "decisions": c.decisions if c else 0,
            "conventions": c.conventions if c else 0, "lessons": c.lessons if c else 0,
        })
    return out


@router.post("/projects/connect", response_model=ConnectJobResponse)
async def connect(body: ConnectBody, engine=Depends(get_engine)):
    pid = body.id.strip()
    if not pid:
        raise HTTPException(status_code=422, detail="project id required")
    meta = registry.all_projects().get(pid)   # includes UI-added overlay (path for re-connect)
    name = body.name or (meta["name"] if meta else pid.replace("-", " ").title())
    host_path = body.path or (meta["path"] if meta else None)
    # Fall back to <root>/<id> for a graph-only re-connect (folder named like the id).
    if not host_path and (guess := registry.project_folder(pid)).is_dir():
        host_path = str(guess)
    description = body.description or (meta["desc"] if meta else
                                      f"{name} is a project connected to Synapse.")
    if not host_path:
        raise HTTPException(status_code=422, detail="path required for an unregistered project")

    folder = _resolve_folder(host_path)   # 404s, naming the roots, when none of them holds it

    actions = _connector.write_files(folder, pid, name)
    blocked = [a for a in actions if a.startswith(("ABORT", "SKIP"))]
    if blocked:
        raise HTTPException(status_code=409, detail="; ".join(blocked))

    # Remember UI-added (unregistered) projects so they persist with their real name/path.
    if pid not in registry.PROJECTS:
        registry.add_connected(pid, name, host_path)

    job_id = uuid.uuid4().hex[:12]
    import time as _time
    _evict_old_jobs()
    _JOBS[job_id] = {"job_id": job_id, "project": pid, "state": "running", "done": 0,
                     "total": 0, "stored": 0, "actions": actions, "entity": None,
                     "_created_at": _time.monotonic()}
    asyncio.create_task(_run_seed(job_id, engine, pid, folder, description,
                                  deep_seed=body.deep_seed))
    return _JOBS[job_id]


@router.get("/projects/connect/{job_id}", response_model=ConnectJobResponse)
async def connect_status(job_id: str):
    _evict_old_jobs()
    job = _JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="unknown job")
    return job


async def _run_seed(job_id: str, engine, project_id: str, folder: Path, description: str,
                    *, deep_seed: bool) -> None:
    """Every graph write a connect performs, off the request thread.

    Seeding the Project entity used to be awaited inline "because it's just one write" — but one
    write is a full LLM extraction. Measured on the connect that exposed this: 73s, against
    nginx's default 60s proxy_read_timeout. The browser got a 504 while the seed went on to
    succeed in the background, which is the worst kind of failure — a false one. The request now
    does filesystem work only, and the job carries everything slow.
    """
    job = _JOBS[job_id]
    scope = f"project_{project_id}"

    async def on_progress(done: int, total: int, stored: int, fact: str | None) -> None:
        job.update(done=done, total=total, stored=stored)
        await bus.publish(KnowledgeEvent(type="project.connect.progress", id=job_id,
                                         scope=scope, summary=fact, done=done, total=total, stored=stored))
        if fact:  # also grow the live graph
            await bus.publish(KnowledgeEvent(type="knowledge.added", scope=scope, summary=fact))

    try:
        job["entity"] = await _connector.seed_entity(engine, project_id, description)
        if deep_seed:
            job.update(**await _connector.deep_seed(engine, project_id, folder, on_progress))
        job["state"] = "done"
        await engine.reader.invalidate_brief(project_id)
        await bus.publish(KnowledgeEvent(type="project.connect.done", id=job_id, scope=scope,
                                         state="done", done=job["total"], total=job["total"], stored=job["stored"]))
    except Exception as exc:  # noqa: BLE001
        logger.exception("seed failed for %s", project_id)
        job.update(state="error", error=str(exc))
        await bus.publish(KnowledgeEvent(type="project.connect.done", id=job_id, scope=scope,
                                         state="error", error=str(exc)))
