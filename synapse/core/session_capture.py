"""Session-lesson auto-capture (Phase: self-seeding).

A Haiku judge reads a session transcript and proposes durable knowledge; high-confidence items are
stored straight away (through the normal write pipeline — so they inherit triage + dedup + hybrid
extraction), and borderline items queue as `PendingCapture` nodes for a quick review in the Curate UI.

R2 (no noise) is enforced by THREE stacked gates: the strict judge prompt, the confidence threshold,
and the write pipeline's own triage/dedup on the auto-stored path. See the design doc for the contract.
"""

from __future__ import annotations

import hashlib
import json
import logging

from pydantic import BaseModel, Field

from synapse.config import settings

logger = logging.getLogger("synapse.capture")

# Durable knowledge types eligible for auto-store (mirrors the schema labels; excludes generic entity).
DURABLE_TYPES = {"decision", "convention", "lesson", "research", "pattern", "tool"}

_JUDGE_SYSTEM = (
    "You read a transcript of a software-engineering session and extract ONLY durable, reusable "
    "knowledge worth remembering for future sessions on this project. STORE: decisions (with the "
    "reasoning), conventions ('always do it this way'), lessons (gotchas, failures, hard-won "
    "best-practices), concluded research findings, reusable patterns, and tool/library choices. "
    "REJECT everything ephemeral: task progress, debugging play-by-play, file edits made, 'ran the "
    "tests', restated user requests, anything trivially project-specific or already-obvious. Quality "
    "over quantity — return an EMPTY array if nothing is genuinely durable. For each kept item write a "
    "self-contained statement (understandable with no other context), pick the best type from "
    f"{sorted(DURABLE_TYPES)}, and rate confidence 0-1 (how clearly durable AND clearly evidenced in the "
    "transcript). Respond with ONLY a JSON array: "
    '[{"content": str, "type": str, "confidence": float, "reason": str}].'
)


class CaptureCandidate(BaseModel):
    content: str
    type: str = "lesson"
    confidence: float = 0.5
    reason: str = ""


class CaptureResult(BaseModel):
    stored: list[str] = Field(default_factory=list)
    pending: list[str] = Field(default_factory=list)
    skipped: bool = False


def _parse_candidates(text: str) -> list[CaptureCandidate]:
    start, end = text.find("["), text.rfind("]")
    try:
        data = json.loads(text[start:end + 1]) if start != -1 and end != -1 else []
    except Exception:  # noqa: BLE001
        return []
    out: list[CaptureCandidate] = []
    for item in data if isinstance(data, list) else []:
        if isinstance(item, dict) and item.get("content"):
            out.append(CaptureCandidate(
                content=str(item["content"]).strip(),
                type=str(item.get("type", "lesson")).lower(),
                confidence=float(item.get("confidence", 0.5)),
                reason=str(item.get("reason", "")),
            ))
    return out


async def default_judge(transcript: str) -> tuple[list[CaptureCandidate], str]:
    """Credit-aware judge: Haiku when the key has credit, else local gemma. Returns (candidates, provider)."""
    from synapse.core.llm_fallback import haiku_or_local

    try:
        text, provider = await haiku_or_local(_JUDGE_SYSTEM, transcript[:24000], max_tokens=1500)
    except Exception as exc:  # noqa: BLE001 — a judge failure means "no captures", never a crash
        logger.warning("capture judge failed: %s", exc)
        return [], "none"
    return _parse_candidates(text), provider


class CaptureEngine:
    def __init__(self, graphiti, judge, remember, *,
                 autostore_threshold: float | None = None, enabled: bool | None = None) -> None:
        self._driver = graphiti.driver
        self._judge = judge          # async: transcript -> list[CaptureCandidate]
        self._remember = remember    # async: (content, knowledge_type, project_id, source, force) -> WriteResult
        self.threshold = autostore_threshold if autostore_threshold is not None else settings.capture_autostore_threshold
        self.enabled = enabled if enabled is not None else settings.capture_enabled

    async def capture(self, project_id: str, session_id: str, transcript: str) -> CaptureResult:
        if not self.enabled or len(transcript.strip()) < 200:
            return CaptureResult(skipped=True)
        result = CaptureResult()
        candidates, provider = await self._judge(transcript)
        # When the judge ran on the weaker LOCAL fallback (credits out), don't auto-store anything —
        # route everything to the review queue for your approval (R2 under degraded judgment).
        degraded = provider == "local"
        for c in candidates:
            if not degraded and c.confidence >= self.threshold and c.type in DURABLE_TYPES:
                r = await self._remember(c.content, knowledge_type=c.type,
                                         project_id=project_id, source="capture")
                # The write pipeline's triage/dedup is gate 3 — only count what actually landed.
                if getattr(r, "outcome", None) is not None and r.outcome.value in ("stored", "contradiction"):
                    result.stored.append(c.content)
            else:
                await self._queue(project_id, session_id, c)
                result.pending.append(c.content)
        logger.info("capture[%s]: %d stored, %d pending", project_id, len(result.stored), len(result.pending))
        return result

    async def _queue(self, project_id: str, session_id: str, c: CaptureCandidate) -> None:
        h = hashlib.sha256(c.content.strip().lower().encode("utf-8")).hexdigest()
        await self._driver.execute_query(
            """
            MERGE (p:PendingCapture {hash: $h})
            ON CREATE SET p.uuid = randomUUID(), p.project_id = $pid, p.content = $content,
                p.type = $type, p.confidence = $conf, p.reason = $reason, p.session_id = $sid,
                p.created_at = datetime(), p.status = 'pending'
            """,
            h=h, pid=project_id, content=c.content, type=c.type,
            conf=c.confidence, reason=c.reason, sid=session_id)

    async def list_pending(self, project_id: str | None = None) -> list[dict]:
        where = "WHERE p.project_id = $pid" if project_id else ""
        res = await self._driver.execute_query(
            f"""
            MATCH (p:PendingCapture) {where}
            RETURN p.uuid AS uuid, p.project_id AS project_id, p.content AS content, p.type AS type,
                   p.confidence AS confidence, p.reason AS reason
            ORDER BY p.created_at DESC LIMIT 100
            """,
            pid=project_id)
        return [dict(r) for r in res.records]

    async def count(self) -> int:
        res = await self._driver.execute_query("MATCH (p:PendingCapture) RETURN count(p) AS n")
        return int(res.records[0]["n"]) if res.records else 0

    async def approve(self, uuid: str) -> dict:
        res = await self._driver.execute_query(
            "MATCH (p:PendingCapture {uuid: $uuid}) RETURN p.content AS content, p.type AS type, "
            "p.project_id AS pid", uuid=uuid)
        if not res.records:
            return {"ok": False, "error": "not found"}
        row = res.records[0]
        await self._remember(row["content"], knowledge_type=row["type"],
                             project_id=row["pid"], source="capture", force=True)
        await self._driver.execute_query("MATCH (p:PendingCapture {uuid: $uuid}) DELETE p", uuid=uuid)
        return {"ok": True, "stored": row["content"]}

    async def dismiss(self, uuid: str) -> dict:
        res = await self._driver.execute_query(
            "MATCH (p:PendingCapture {uuid: $uuid}) DELETE p RETURN count(*) AS n", uuid=uuid)
        return {"ok": True}


def build_capture_engine(graphiti, remember) -> CaptureEngine:
    return CaptureEngine(graphiti, default_judge, remember)
