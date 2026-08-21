"""Retrieval feedback — which knowledge actually earns its place (roadmap item 14).

Research §6. Nothing recorded whether a recalled fact was useful, so every ranking decision in this
system has been tuned by intuition against a 16-case golden set. This is the route to tuning by
evidence.

**What this deliberately does NOT claim.** We cannot observe whether an agent *used* an injected
fact — that would mean reading its reasoning. Claiming a "used" signal would be a fiction, and a
fiction wired into ranking is worse than no signal. So only three things are recorded, all of them
directly observable:

``recalled_n`` / ``last_recalled_at``
    Impressions: how often a fact was actually served to a consumer, and when. A fact served 50
    times is load-bearing; one never served in months is dead weight regardless of how good it looks.

``corrected_n`` / ``last_corrected_at``
    How often a fact was later ``update``d or ``forgot``ten. This is the strongest available quality
    signal, because it is an explicit human/agent judgement that the fact was wrong.

Together those give a usable ratio: **served often and never corrected = trustworthy; corrected
shortly after being served = suspect.**

**Impressions are opt-in per call**, which matters more than it sounds. If every read counted, the
eval harness would inflate the counters of exactly the facts it measures, UI browsing would look
like agent usage, and the signal would be self-referential within a day. Only a real consumption —
an agent recall, the ``UserPromptSubmit`` hook — passes ``feedback=True``.

Counters live on the fact edge rather than in separate event nodes: the volume is one property
update per served fact (a few thousand a day at most), and the consumer of this signal is the
ranker, which already loads the edge. Per-session detail is provenance's job
(``synapse/core/provenance.py``), not this module's.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field

logger = logging.getLogger("synapse.feedback")


class FactFeedback(BaseModel):
    """Observed usage of one fact."""

    uuid: str
    fact: str = ""
    scope: str = ""
    recalled_n: int = 0
    corrected_n: int = 0

    @property
    def is_suspect(self) -> bool:
        """Corrected at least once despite being served — an explicit judgement that it was wrong."""
        return self.corrected_n > 0

    @property
    def is_dead_weight(self) -> bool:
        """Never served to any consumer, so it has never earned its place in the corpus."""
        return self.recalled_n == 0


class FeedbackSummary(BaseModel):
    """Corpus-level view of what retrieval is actually delivering."""

    total_facts: int = 0
    ever_recalled: int = 0
    never_recalled: int = 0
    total_impressions: int = 0
    corrected_facts: int = 0
    most_recalled: list[FactFeedback] = Field(default_factory=list)
    suspect: list[FactFeedback] = Field(default_factory=list)

    @property
    def coverage(self) -> float:
        """Fraction of the corpus that has ever been served. Low = the corpus is mostly unread."""
        return (self.ever_recalled / self.total_facts) if self.total_facts else 0.0
