"""Two-judge relevance grading, with the statistics that make the grades defensible.

An LLM judge is an instrument, and an instrument that is never calibrated is a source of numbers,
not evidence. The public record here is unkind: an independent audit of LoCoMo found its judge
accepted **62.8%** of deliberately wrong answers, which means every score reported against it sat
below the noise floor of its own grader. A single unvalidated judge would put this project in the
same position.

So three things, none of which are optional:

1. **Two judges from different model families.** Claude and a local gemma. Judges show measurable
   self-preference for their own family's text, and two calls to the same model measure that
   model's consistency, not the grade's validity. They are never allowed to fall back to each
   other — a judge that silently becomes the other judge destroys the independence the protocol
   exists for, so a failed judge is recorded as an abstention.
2. **Cohen's kappa between them**, reported next to any score. Kappa corrects for the agreement
   two graders reach by chance, which matters here because relevance grades are heavily skewed
   toward 0 and raw agreement therefore looks impressive for free.
3. **An adversarial probe.** A sample of judgements is repeated with a plausible-but-wrong fact
   substituted. The share the judge accepts anyway is the floor below which score differences mean
   nothing, and it belongs in the report beside the headline number.

Grades are 0/1/2, not 1-10. A coarse scale that two models can apply consistently is worth more
than a fine one they cannot.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections import Counter
from typing import Any, NamedTuple

from synapse.config import settings

logger = logging.getLogger("synapse.eval.judge")

#: Deliberately terse. A long rubric invites the model to reason its way to generosity.
RUBRIC = """You grade whether a retrieved fact answers a developer's question about their own codebase.

Grade strictly:
2 = directly answers the question, on its own
1 = related and useful context, but does not answer it
0 = unrelated, or merely shares vocabulary with the question

Reply with a single digit: 0, 1, or 2. No other text."""

#: Grade returned when a judge errored or produced something ungradeable. Kept out of the
#: agreement statistics rather than being silently coerced to 0 — an abstention is missing data,
#: and treating it as "irrelevant" would quietly bias every score downward.
ABSTAIN = -1

_DIGIT = re.compile(r"[012]")


class Judgement(NamedTuple):
    uuid: str
    query: str
    grades: dict[str, int]      # judge name -> grade (or ABSTAIN)

    @property
    def agreed(self) -> bool:
        real = [g for g in self.grades.values() if g != ABSTAIN]
        return len(real) > 1 and len(set(real)) == 1

    @property
    def consensus(self) -> int:
        """Both judges' grade when they agree; the lower when they do not.

        Taking the lower on disagreement is the conservative choice: it under-counts relevance, so
        a retrieval score built on it is a floor rather than a flattering estimate.
        """
        real = [g for g in self.grades.values() if g != ABSTAIN]
        return min(real) if real else ABSTAIN


def parse_grade(text: str) -> int:
    """First 0/1/2 in the reply, or ABSTAIN. Models add prose despite being told not to."""
    if not text:
        return ABSTAIN
    m = _DIGIT.search(text.strip())
    return int(m.group()) if m else ABSTAIN


def make_anthropic_judge(model: str) -> Any:
    """A grader pinned to a specific Anthropic model.

    Judge strength is a lever on agreement, and agreement is what decides whether a judged number
    means anything. Haiku is the right default for volume; a hard question deserves a better
    reader, and paying for one is cheaper than publishing a number two weak judges disagreed about.
    """

    async def grade(query: str, fact: str, rubric: str = RUBRIC) -> int:
        return await _grade_anthropic(query, fact, rubric, model=model)

    return grade


async def _grade_anthropic(
    query: str, fact: str, rubric: str = RUBRIC, *, model: str | None = None
) -> int:
    from anthropic import AsyncAnthropic

    client = AsyncAnthropic(api_key=settings.anthropic_api_key)
    msg = await client.messages.create(
        model=model or settings.triage_model,
        max_tokens=8,
        temperature=0,
        system=rubric,
        messages=[{"role": "user", "content": f"{query}\n\n{fact}\n\nGrade:"}],
    )
    text = "".join(
        getattr(b, "text", "") for b in msg.content if getattr(b, "type", None) == "text"
    )
    return parse_grade(text)


async def _grade_local(query: str, fact: str, rubric: str = RUBRIC) -> int:
    from openai import AsyncOpenAI

    client = AsyncOpenAI(base_url=settings.ollama_base_url, api_key="ollama")
    resp = await client.chat.completions.create(
        model=settings.local_extraction_model,
        max_tokens=8,
        temperature=0,
        messages=[
            {"role": "system", "content": rubric},
            {"role": "user", "content": f"{query}\n\n{fact}\n\nGrade:"},
        ],
    )
    return parse_grade(resp.choices[0].message.content or "")


#: name -> grader. Two families on purpose; see the module docstring.
DEFAULT_JUDGES: dict[str, Any] = {"claude": _grade_anthropic, "gemma": _grade_local}


async def judge_one(
    query: str, uuid: str, fact: str, judges: dict[str, Any] | None = None,
    rubric: str | None = None,
) -> Judgement:
    """Grade one (query, fact) pair with every judge. A judge that raises abstains.

    ``rubric`` overrides the relevance rubric so the same two-family protocol — different model
    families, no fallback between them, kappa reported — can be pointed at a different question.
    It is the protocol that carries the credibility, not the particular question it is asked.
    """
    judges = judges or DEFAULT_JUDGES
    names = list(judges)
    kwargs = {"rubric": rubric} if rubric is not None else {}
    results = await asyncio.gather(
        *(judges[n](query, fact, **kwargs) for n in names), return_exceptions=True
    )
    grades: dict[str, int] = {}
    for name, res in zip(names, results):
        if isinstance(res, BaseException):
            logger.warning("judge %r failed: %s", name, str(res)[:160])
            grades[name] = ABSTAIN
        else:
            grades[name] = res
    return Judgement(uuid=uuid, query=query, grades=grades)


def cohens_kappa(a: list[int], b: list[int]) -> float | None:
    """Chance-corrected agreement between two graders over paired grades.

    Pairs where either judge abstained are dropped, not imputed. Returns None when fewer than two
    gradeable pairs survive, and 1.0 when both graders were perfectly constant and identical —
    the degenerate case where the chance-agreement denominator vanishes.
    """
    pairs = [(x, y) for x, y in zip(a, b) if x != ABSTAIN and y != ABSTAIN]
    n = len(pairs)
    if n < 2:
        return None
    observed = sum(1 for x, y in pairs if x == y) / n
    ca, cb = Counter(x for x, _ in pairs), Counter(y for _, y in pairs)
    expected = sum((ca[c] / n) * (cb[c] / n) for c in set(ca) | set(cb))
    if expected >= 1.0:
        # Both graders gave the same single grade to everything. There is no chance-corrected
        # signal to extract; reporting perfect agreement is more honest than dividing by zero.
        return 1.0 if observed == 1.0 else 0.0
    return round((observed - expected) / (1 - expected), 3)


def agreement_report(judgements: list[Judgement], judge_a: str, judge_b: str) -> dict[str, Any]:
    """The block that has to sit next to any judged score."""
    a = [j.grades.get(judge_a, ABSTAIN) for j in judgements]
    b = [j.grades.get(judge_b, ABSTAIN) for j in judgements]
    gradeable = [(x, y) for x, y in zip(a, b) if x != ABSTAIN and y != ABSTAIN]
    exact = sum(1 for x, y in gradeable if x == y)
    return {
        "n": len(judgements),
        "n_gradeable": len(gradeable),
        "abstentions": sum(1 for j in judgements for g in j.grades.values() if g == ABSTAIN),
        "exact_agreement": round(exact / len(gradeable), 3) if gradeable else None,
        "cohens_kappa": cohens_kappa(a, b),
        "grade_distribution": {
            judge_a: dict(Counter(x for x in a if x != ABSTAIN)),
            judge_b: dict(Counter(y for y in b if y != ABSTAIN)),
        },
    }


async def adversarial_probe(
    pairs: list[tuple[str, str]], judges: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Grade deliberately wrong facts and report how many each judge accepted anyway.

    *pairs* is (query, decoy_fact) where the decoy is plausible prose that does not answer the
    query — the same shape as the LoCoMo audit that found a 62.8% acceptance rate. Anything graded
    1 or 2 is an acceptance. The resulting rate is the noise floor for every other number in the
    report: score differences smaller than it are not measurements.
    """
    judgements = await asyncio.gather(
        *(judge_one(q, f"decoy-{i}", fact, judges) for i, (q, fact) in enumerate(pairs))
    )
    per_judge: dict[str, dict[str, Any]] = {}
    for name in (judges or DEFAULT_JUDGES):
        graded = [j.grades.get(name, ABSTAIN) for j in judgements]
        real = [g for g in graded if g != ABSTAIN]
        accepted = sum(1 for g in real if g >= 1)
        per_judge[name] = {
            "n": len(real),
            "accepted": accepted,
            "acceptance_rate": round(accepted / len(real), 3) if real else None,
        }
    return {"probes": len(pairs), "per_judge": per_judge}
