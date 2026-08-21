"""Validate the structural invalidation guard against an INDEPENDENT judgement.

    python -m scripts.validate_invalidation_guard --sample 30

Why this exists
---------------
``scripts/audit_invalidations.py`` reports two numbers — how often the structural guard would block
a retirement, and an upper bound on how often it would block a real supersession. Both are computed
with the same function, ``could_replace``, fed different identity keys. That is not validation. It
is one heuristic agreeing with a relaxed copy of itself, and if ``could_replace``'s notion of "same
relationship" is wrong, both numbers move together and neither notices.

So this asks a different kind of witness. For a stratified sample of retirements it shows two
judges — from different model families, neither able to fall back to the other — the retired fact
and the facts the attributed write actually extracted, and asks in plain language whether the new
facts make the old one no longer true. The judges never see the structural verdict, so the
comparison is blind by construction.

That yields a confusion matrix between a structural rule and a semantic reading, which is what
"is the guard right" actually requires. Agreement is not proof — two judges can be wrong together,
which is why their kappa is reported next to the result — but it is evidence from outside the
rule being tested, and that is the property the audit script cannot have.

Read-only against Neo4j. Sends fact text to the judges, so it costs a small number of LLM calls;
the local judge is free and the cloud judge is Haiku.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import random
import re
import sys
from datetime import datetime
from typing import Any

from synapse.config import settings
from synapse.core.consolidation_engine import could_replace
from synapse.eval.judge import (
    ABSTAIN,
    _grade_local,
    agreement_report,
    cohens_kappa,
    judge_one,
    make_anthropic_judge,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

WINDOW_SECONDS = 30
#: Fixed so a rerun samples the same retirements and the numbers are comparable across runs.
SEED = 20260820
#: How many extracted facts to show the judge. Enough to see the write's intent, few enough that
#: the judge is reading rather than skimming.
MAX_CONTEXT_FACTS = 12

#: Deliberately binary and deliberately narrow. The question is NOT "are these related" — related
#: is what got us here. It is whether the new information makes the old statement false.
#:
#: The worked examples are load-bearing. A first version without them produced Cohen's kappa of
#: 0.393 between the two judges, which is too little agreement to carry any number: the judges
#: were reading "supersedes" differently, one treating a topical update as a replacement and the
#: other requiring a direct conflict. Pinning the boundary with cases is what a rubric is for.
SUPERSESSION_RUBRIC = """You are auditing a knowledge graph. An old fact was retired at the moment some new facts were written. Decide whether retiring it was CORRECT.

Answer 1 only if one of the NEW facts states something that makes the OLD fact false — a direct conflict, or a replacement of the same value/relationship.

Answer 0 if the new facts are about a different relationship, add detail without contradicting, or merely mention the same entities.

Worked examples:

OLD: "The gateway is published on host port 4003."
NEW: "The gateway is published on host port 4005."
-> 1  (same relationship, different value)

OLD: "TimescaleDB is used for market data storage."
NEW: "The gateway uses clientIds 17-21 for isolation."
-> 0  (different relationship entirely; same project is not enough)

OLD: "Sessions expire after 30 minutes."
NEW: "Sessions are stored in Redis."
-> 0  (adds detail, contradicts nothing)

OLD: "The project uses Poetry for dependency management."
NEW: "The project migrated from Poetry to uv."
-> 1  (explicit replacement)

OLD: "Retries use exponential backoff."
NEW: "Retries use exponential backoff with a five-attempt cap."
-> 0  (a refinement, not a contradiction — the old statement is still true)

Reply with a single digit: 0 or 1. No other text."""

#: Judge only where attribution is unambiguous. In a 20-edge cohort the write's intent is diffuse
#: and the prompt cannot show every fact, so a disagreement between judges may be about what they
#: were shown rather than about the retirement. Small cohorts remove that confound; the cohort-size
#: stratification separately shows the block rate is HIGHER here, so this is not cherry-picking an
#: easy stratum.
#: Raised from 8 to 15 because at 8 only 7 retirements the guard ALLOWS were available, and a
#: missed-damage rate off n=7 has a 95% interval of [0%, 35%] — uninformative. The allow side is
#: the scarce stratum and it is precisely where a false negative hides, so the limit is set by what
#: makes that stratum measurable, not by what makes the block side tidy.
MAX_COHORT_FOR_JUDGING = 15

RETIRED = """
MATCH (a)-[r:RELATES_TO]->(b)
WHERE r.expired_at IS NOT NULL
RETURN r.uuid AS uuid, r.fact AS fact, a.uuid AS src, b.uuid AS dst, r.name AS name,
       r.group_id AS scope, toString(r.expired_at) AS expired_at
"""

ALL_EDGES = """
MATCH (a)-[r:RELATES_TO]->(b)
RETURN r.uuid AS uuid, r.fact AS fact, a.uuid AS src, b.uuid AS dst, r.name AS name,
       r.group_id AS scope, toString(r.created_at) AS created_at
"""


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson interval — correct at the small n these strata actually have."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def _parse(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00").split("[")[0])
    except ValueError:
        return None


async def _load() -> tuple[list[dict], list[dict]]:
    from neo4j import AsyncGraphDatabase

    driver = AsyncGraphDatabase.driver(
        os.environ["NEO4J_URI"], auth=("neo4j", os.environ["NEO4J_PASSWORD"])
    )
    try:
        retired_rows, _, _ = await driver.execute_query(RETIRED)
        all_rows, _, _ = await driver.execute_query(ALL_EDGES)
    finally:
        await driver.close()
    retired = [d for d in (dict(r, _t=_parse(r["expired_at"])) for r in retired_rows) if d["_t"]]
    created = [d for d in (dict(c, _t=_parse(c["created_at"])) for c in all_rows) if d["_t"]]
    return retired, created


def attribute(retired: list[dict], created: list[dict]) -> list[dict]:
    """Pair each retirement with its cohort and the structural verdict."""
    out = []
    for r in retired:
        cohort = [
            c for c in created
            if c["uuid"] != r["uuid"] and c["scope"] == r["scope"]
            and abs((c["_t"] - r["_t"]).total_seconds()) <= WINDOW_SECONDS
        ]
        if not cohort:
            continue
        old = (r["src"], r["dst"], r["name"])
        allowed = any(could_replace((c["src"], c["dst"], c["name"]), old) for c in cohort)
        out.append({"retired": r, "cohort": cohort, "structural_allows": allowed})
    return out


def build_prompt(item: dict) -> tuple[str, str]:
    """(old, new) halves of the judge prompt. Carries no hint of the structural verdict."""
    old = f"OLD FACT (was retired):\n{item['retired']['fact']}"
    # Smallest cohorts first: an unambiguous attribution should be the easiest to read.
    facts = [c["fact"] for c in item["cohort"] if c.get("fact")][:MAX_CONTEXT_FACTS]
    new = "NEW FACTS (written by the update that retired it):\n" + "\n".join(
        f"- {f}" for f in facts
    )
    return old, new


HUMAN_HEADER = """# Invalidation labelling — {n} cases

Two models could not agree on this task (Cohen's kappa 0.35 across two rubrics and two judge
pairings), so it needs a human. That is the whole reason this file exists.

**For each case below, put 0 or 1 after `ANSWER:` and save the file.**

  1 = retiring the old fact was CORRECT — one of the new facts makes it false
      (a direct conflict, or a replacement of the same value or relationship)

  0 = retiring it was WRONG — the old fact should have been kept
      (the new facts are about a different relationship, add detail without contradicting,
       or merely mention the same things)

Being on a related topic, sharing words, or naming the same entity is NOT enough.
There has to be a genuine conflict or replacement.

Worked examples:

  OLD: The gateway is published on host port 4003.
  NEW: The gateway is published on host port 4005.
  -> 1   same relationship, different value

  OLD: TimescaleDB is used for market data storage.
  NEW: The gateway uses clientIds 17-21 for isolation.
  -> 0   different relationship; same project is not enough

  OLD: Retries use exponential backoff.
  NEW: Retries use exponential backoff with a five-attempt cap.
  -> 0   a refinement, not a contradiction — the old statement is still true

Leave `ANSWER:` blank to skip a case you cannot judge. Skipping is better than guessing:
an unlabelled case is missing data, a guessed one is wrong data.

The cases are shuffled and carry no hint of what the guard decided, so your labels stay
independent of the thing they are measuring. Read them back with:

    python -m scripts.validate_invalidation_guard --labels {path}

---
"""

CASE_TEMPLATE = """
## {idx}

OLD FACT (was retired):
    {old}

NEW FACTS (written by the update that retired it):
{new}

ANSWER:

---
"""


def _write_labelling_file(path: str, chosen: list[dict], key_path: str,
                          model_grades: list[dict[str, int]] | None) -> None:
    lines = [HUMAN_HEADER.format(n=len(chosen), path=path)]
    key: dict[str, Any] = {"window_seconds": WINDOW_SECONDS, "seed": SEED, "cases": {}}
    for idx, item in enumerate(chosen, start=1):
        facts = [c["fact"] for c in item["cohort"] if c.get("fact")][:MAX_CONTEXT_FACTS]
        lines.append(CASE_TEMPLATE.format(
            idx=idx,
            old=item["retired"]["fact"],
            new="\n".join(f"    - {f}" for f in facts) or "    (none extracted)",
        ))
        # The key holds what the human must not see. Separate file, written at the same time.
        key["cases"][str(idx)] = {
            "uuid": item["retired"]["uuid"],
            "structural_allows": item["structural_allows"],
            "cohort_size": len(item["cohort"]),
            "models": (model_grades[idx - 1] if model_grades else {}),
        }
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("".join(lines))
    with open(key_path, "w", encoding="utf-8") as fh:
        json.dump(key, fh, indent=2)


_CASE_RE = re.compile(r"^##\s+(\d+)\s*$")
_ANSWER_RE = re.compile(r"^ANSWER:\s*([01])?\s*$")


def _read_labels(path: str) -> dict[str, int]:
    """Parse `## N` / `ANSWER: X` pairs. A blank answer is a deliberate skip, not a zero."""
    labels: dict[str, int] = {}
    current: str | None = None
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            m = _CASE_RE.match(line)
            if m:
                current = m.group(1)
                continue
            a = _ANSWER_RE.match(line)
            if a and current is not None:
                if a.group(1) is not None:
                    labels[current] = int(a.group(1))
                current = None
    return labels


def _report_against_labels(key_path: str, labels: dict[str, int]) -> int:
    with open(key_path, encoding="utf-8") as fh:
        key = json.load(fh)
    cases = key["cases"]
    total = len(cases)
    if not labels:
        print(f"[error] no answers found in the labels file ({total} cases were exported).")
        return 1

    matrix = {("allow", 1): 0, ("allow", 0): 0, ("block", 1): 0, ("block", 0): 0}
    for idx, verdict in labels.items():
        meta = cases.get(idx)
        if meta is None:
            continue
        matrix[("allow" if meta["structural_allows"] else "block", verdict)] += 1

    print(f"labelled {len(labels)} of {total} cases ({total - len(labels)} skipped)\n")
    print("confusion matrix (your labels vs the structural rule):")
    print(f"{'':<10}{'you: correctly retired':>26}{'you: should have kept':>24}")
    print(f"{'ALLOW':<10}{matrix[('allow', 1)]:>26}{matrix[('allow', 0)]:>24}")
    print(f"{'BLOCK':<10}{matrix[('block', 1)]:>26}{matrix[('block', 0)]:>24}")

    fp, tp = matrix[("block", 1)], matrix[("block", 0)]
    tn, fn = matrix[("allow", 1)], matrix[("allow", 0)]
    if tp + fp:
        lo, hi = wilson(fp, tp + fp)
        print(f"\n  WRONG-BLOCK rate  : {fp}/{tp + fp} = {fp / (tp + fp):.1%} "
              f"[{lo:.1%}, {hi:.1%}]   (guard blocks a retirement you call correct)")
    if tn + fn:
        lo, hi = wilson(fn, tn + fn)
        print(f"  MISSED-DAMAGE rate: {fn}/{tn + fn} = {fn / (tn + fn):.1%} "
              f"[{lo:.1%}, {hi:.1%}]   (guard allows one you call wrong)")

    # How did the models do against you? This is what says whether the kappa 0.35 was the models
    # being weak or the task being genuinely underdetermined.
    for name in ("sonnet", "gemma"):
        pairs = [(labels[i], cases[i]["models"].get(name))
                 for i in labels if cases.get(i, {}).get("models", {}).get(name) is not None]
        pairs = [(h, m) for h, m in pairs if m in (0, 1)]
        if len(pairs) >= 2:
            agree = sum(1 for h, m in pairs if h == m) / len(pairs)
            k = cohens_kappa([h for h, _ in pairs], [m for _, m in pairs])
            print(f"  {name:<7} vs you: {agree:.1%} agreement, kappa {k} (n={len(pairs)})")
    return 0


async def main(sample: int, export: str | None, labels_path: str | None) -> int:
    if labels_path:
        key_path = labels_path + ".key.json"
        if not os.path.exists(key_path):
            print(f"[error] {key_path} not found — it is written next to the labels file by "
                  "--export and holds the verdicts you were not shown.")
            return 2
        return _report_against_labels(key_path, _read_labels(labels_path))

    if not os.environ.get("NEO4J_URI") or not os.environ.get("NEO4J_PASSWORD"):
        print("[error] NEO4J_URI and NEO4J_PASSWORD must be set.")
        return 2

    retired, created = await _load()
    items = attribute(retired, created)
    print(f"attributed retirements: {len(items)}  "
          f"(structural: {sum(i['structural_allows'] for i in items)} allow / "
          f"{sum(not i['structural_allows'] for i in items)} block)")

    items = [i for i in items if len(i["cohort"]) <= MAX_COHORT_FOR_JUDGING]
    allowed = [i for i in items if i["structural_allows"]]
    blocked = [i for i in items if not i["structural_allows"]]
    print(f"judgeable (cohort <= {MAX_COHORT_FOR_JUDGING}): {len(items)}  "
          f"({len(allowed)} allow / {len(blocked)} block)")

    rng = random.Random(SEED)
    # Stratified: an unstratified sample would be ~76% blocked and tell us almost nothing about
    # the allow side, which is exactly where a false negative would hide.
    pick_a = rng.sample(allowed, min(sample, len(allowed)))
    pick_b = rng.sample(blocked, min(sample, len(blocked)))
    chosen = pick_a + pick_b
    rng.shuffle(chosen)
    print(f"judging {len(chosen)} retirements ({len(pick_a)} allow, {len(pick_b)} block) "
          f"x 2 judges...\n")

    # Sonnet rather than Haiku on the cloud side: this is a harder question than relevance
    # grading, and the first run's kappa of 0.393 said the instrument, not the guard, was the
    # limiting factor. Still two different families, still no fallback between them.
    judges = {
        "sonnet": make_anthropic_judge(settings.extraction_model),
        "gemma": _grade_local,
    }
    judgements = []
    for item in chosen:
        old, new = build_prompt(item)
        judgements.append(
            await judge_one(old, item["retired"]["uuid"], new, judges=judges,
                            rubric=SUPERSESSION_RUBRIC)
        )

    agree = agreement_report(judgements, "sonnet", "gemma")
    print("judge agreement (the instrument, before any result):")
    for key in ("n", "n_gradeable", "abstentions", "exact_agreement", "cohens_kappa"):
        print(f"  {key:<18} {agree.get(key)}")

    if export:
        # The model grades ride along in the key file so the read-back can report how each model
        # did against the human. That is what distinguishes "the models are weak" from "the task
        # is underdetermined" — and it costs nothing extra, the calls were already made.
        _write_labelling_file(export, chosen, export + ".key.json",
                              [dict(j.grades) for j in judgements])
        print(f"\nWrote {export}  ({len(chosen)} cases to label)")
        print(f"Wrote {export}.key.json  (the verdicts you must not see while labelling)")
        print("\nOpen the first file in any editor, put 0 or 1 after each ANSWER:, save, then run:")
        print(f"  python -m scripts.validate_invalidation_guard --labels {export}")
        return 0

    # Consensus: both judges must agree the retirement was correct for it to count as a genuine
    # supersession. Disagreement falls to "not clearly a supersession", which biases AGAINST the
    # guard — it inflates the apparent false-negative rate rather than flattering it.
    matrix = {("allow", 1): 0, ("allow", 0): 0, ("block", 1): 0, ("block", 0): 0}
    undecided = 0
    for item, j in zip(chosen, judgements):
        grades = [g for g in j.grades.values() if g != ABSTAIN]
        if len(grades) < 2:
            undecided += 1
            continue
        verdict = 1 if all(g == 1 for g in grades) else 0
        matrix[("allow" if item["structural_allows"] else "block", verdict)] += 1

    print(f"\nconfusion matrix (judge consensus vs structural rule), {undecided} undecided:")
    print(f"{'':<10}{'judges: real supersession':>28}{'judges: not':>14}")
    print(f"{'ALLOW':<10}{matrix[('allow', 1)]:>28}{matrix[('allow', 0)]:>14}")
    print(f"{'BLOCK':<10}{matrix[('block', 1)]:>28}{matrix[('block', 0)]:>14}")

    tp = matrix[("block", 0)]   # blocked something the judges also call collateral damage
    fp = matrix[("block", 1)]   # blocked a real supersession — the cost
    tn = matrix[("allow", 1)]   # allowed a real supersession
    fn = matrix[("allow", 0)]   # allowed collateral damage through

    n_block = tp + fp
    n_allow = tn + fn
    if n_block:
        lo, hi = wilson(fp, n_block)
        print(f"\n  of retirements the guard BLOCKS, judges call {fp}/{n_block} "
              f"genuine supersessions -> wrong-block rate {fp / n_block:.1%} "
              f"[{lo:.1%}, {hi:.1%}]")
    if n_allow:
        lo, hi = wilson(fn, n_allow)
        print(f"  of retirements the guard ALLOWS, judges call {fn}/{n_allow} "
              f"collateral damage    -> missed-damage rate {fn / n_allow:.1%} "
              f"[{lo:.1%}, {hi:.1%}]")

    kappa = agree.get("cohens_kappa")
    if kappa is None or kappa < 0.4:
        print(
            f"\n[warning] judge kappa is {kappa}. Below ~0.4 the judges are not agreeing well "
            "enough to carry a headline number — treat everything above as indicative and fix "
            "the rubric before quoting it."
        )
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Validate the invalidation guard against judges")
    p.add_argument("--sample", type=int, default=30, help="Retirements per stratum.")
    p.add_argument("--export", metavar="PATH", default=None,
                   help="Write a file for a human to label, plus PATH.key.json beside it.")
    p.add_argument("--labels", metavar="PATH", default=None,
                   help="Read a labelled file back and report the confusion matrix.")
    args = p.parse_args()
    raise SystemExit(asyncio.run(main(args.sample, args.export, args.labels)))
