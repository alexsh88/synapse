"""Session-capture routes — the hook posts transcripts here; the UI reviews the pending queue.

POST /capture                  — hook trigger: judge a transcript, auto-store or queue (see design).
GET  /captures                 — pending captures awaiting review (optional ?project= and ?limit=).
GET  /captures/count           — pending count (UI badge).
POST /captures/{uuid}/approve  — store the pending capture and remove it from the queue.
POST /captures/{uuid}/dismiss  — drop the pending capture.
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from synapse.api.deps import get_engine
from synapse.models.api import CaptureAccepted, CaptureCountResponse

logger = logging.getLogger("synapse.api.capture")
router = APIRouter(tags=["capture"])


class CaptureBody(BaseModel):
    project_id: str
    session_id: str = ""
    transcript: str


async def _run_capture(engine, body: CaptureBody) -> None:
    try:
        await engine.capture.capture(body.project_id, body.session_id, body.transcript)
    except Exception:  # noqa: BLE001
        # Dead-letter: attempt to queue via the engine pending-capture mechanism so the
        # transcript is not silently lost. _queue is internal to CaptureEngine — we call
        # capture with a tiny stub candidate so the engine’s own _queue path runs if
        # the failure was upstream of that step.
        transcript_len = len(body.transcript)
        logger.error(
            "session capture failed for project=%s session=%s transcript_len=%d;"
            " attempting dead-letter queue",
            body.project_id,
            body.session_id or "(none)",
            transcript_len,
        )
        # Try the engine’s pending-capture queue path directly (CaptureEngine._queue).
        # This preserves the transcript as a PendingCapture so the operator can
        # approve/dismiss it from the Curate UI rather than losing it entirely.
        try:
            from synapse.core.session_capture import CaptureCandidate
            candidate = CaptureCandidate(
                content=body.transcript[:4000],
                type="lesson",
                confidence=0.0,
                reason="dead-letter: capture pipeline failed",
            )
            await engine.capture._queue(body.project_id, body.session_id, candidate)
            logger.info(
                "dead-letter queued for project=%s transcript_len=%d",
                body.project_id, transcript_len,
            )
        except Exception:  # noqa: BLE001
            logger.error(
                "dead-letter queue also failed for project=%s transcript_len=%d"
                " — transcript lost; replay from logs",
                body.project_id, transcript_len,
            )


@router.post("/capture", response_model=CaptureAccepted)
async def capture(body: CaptureBody, engine=Depends(get_engine)):
    # Process in the background: auto-store goes through the (possibly slow, local) write pipeline,
    # so we must not make the fire-and-forget hook wait (its timeout would kill the request).
    if len(body.transcript.strip()) < 200:
        return {"accepted": False, "reason": "transcript too short"}
    asyncio.create_task(_run_capture(engine, body))
    return {"accepted": True}


@router.get("/captures")
async def list_captures(
    project: str | None = None,
    limit: int = Query(default=20, ge=1, le=200),
    engine=Depends(get_engine),
):
    results = await engine.capture.list_pending(project)
    return results[:limit]


@router.get("/captures/count", response_model=CaptureCountResponse)
async def captures_count(engine=Depends(get_engine)):
    return {"count": await engine.capture.count()}


@router.post("/captures/{uuid}/approve")
async def approve_capture(uuid: str, engine=Depends(get_engine)):
    result = await engine.capture.approve(uuid)
    if not result.get("ok"):
        raise HTTPException(status_code=404, detail=result.get("error", "not found"))
    return result


@router.post("/captures/{uuid}/dismiss")
async def dismiss_capture(uuid: str, engine=Depends(get_engine)):
    return await engine.capture.dismiss(uuid)
