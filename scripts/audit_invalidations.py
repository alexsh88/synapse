"""Measure how many of this graph's edge retirements were structurally unjustifiable.

    python -m scripts.audit_invalidations

Applies the structural test from ``synapse.core.consolidation_engine.could_replace`` — the same
rule proposed upstream in getzep/graphiti#1729 — retrospectively to every edge already retired.

For each retired edge, find the write that retired it, then ask whether any edge that write
CREATED could plausibly have replaced it. If nothing qualifies, the retirement was collateral
damage: a fact was silently expired by a write that made no claim about that relationship.

That alone measures only false positives. The script then splits the BLOCKED set again, matching
on entity NAME instead of uuid, to bound the other direction. For a uuid-based structural test the
dominant false-negative mechanism is entity-identity drift: if extraction creates a second node for
the same real-world entity, a genuine supersession has no shared uuid and gets blocked. A
retirement that stays blocked even under name matching is confident collateral damage; one that
flips is a candidate wrong block. Name matching is deliberately over-permissive, so the flipped
count is an UPPER bound on that failure mode — which is what makes "the guard is nearly free"
a defensible claim rather than a hopeful one.

Two methodology notes, because the number this prints is only as good as they are:

* Attribution uses ``expired_at`` (TRANSACTION time — when the write ran), never ``invalid_at``
  (VALID time, which Graphiti may backdate to when the fact stopped being true). Joining on valid
  time pairs each retirement with whatever happened to be written on an unrelated date, and
  collapses the decidable set by roughly 4x.
* The originating write is approximated by edges created within a window of that instant, in the
  same ``group_id``. The window is SWEPT rather than chosen: if the answer moves with the window,
  the approximation is driving it and the number should not be quoted.

Read-only. Prints counts only — never fact text, never credentials.
"""

from __future__ import annotations

import asyncio
import math
import os
import re
import sys
from datetime import datetime

from synapse.core.consolidation_engine import could_replace

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

RETIRED = """
MATCH (a)-[r:RELATES_TO]->(b)
WHERE r.expired_at IS NOT NULL
RETURN r.uuid AS uuid, a.uuid AS src, b.uuid AS dst, r.name AS name,
       a.name AS src_name, b.name AS dst_name,
       r.group_id AS scope, toString(r.expired_at) AS expired_at,
       r.invalidation_reverted_from AS reverted
"""

ALL_EDGES = """
MATCH (a)-[r:RELATES_TO]->(b)
RETURN r.uuid AS uuid, a.uuid AS src, b.uuid AS dst, r.name AS name,
       a.name AS src_name, b.name AS dst_name,
       r.group_id AS scope, toString(r.created_at) AS created_at
"""

WINDOWS_SECONDS = [5, 30, 60]

#: The window the name-identity split reports against. 30s is where coverage is high (91% of
#: retirements decidable) without widening far enough to sweep in an unrelated write.
PRIMARY_WINDOW = 30

_PUNCT = re.compile(r"[^a-z0-9 ]")
_ARTICLES = re.compile(r"\b(the|a|an)\b")


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson interval. Correct at the small n these strata actually have, unlike the
    normal approximation, which produces intervals running past 0 or 1."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) * z / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def _entity_key(name: str | None) -> str:
    """Fold an entity name for identity comparison — case, punctuation and articles drift."""
    folded = _ARTICLES.sub(" ", _PUNCT.sub(" ", (name or "").lower()))
    return " ".join(folded.split())


def _parse(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00").split("[")[0])
    except ValueError:
        return None


async def main() -> int:
    uri = os.environ.get("NEO4J_URI")
    password = os.environ.get("NEO4J_PASSWORD")
    if not uri or not password:
        print("[error] NEO4J_URI and NEO4J_PASSWORD must be set.")
        return 2

    from neo4j import AsyncGraphDatabase

    driver = AsyncGraphDatabase.driver(uri, auth=("neo4j", password))
    try:
        retired_rows, _, _ = await driver.execute_query(RETIRED)
        all_rows, _, _ = await driver.execute_query(ALL_EDGES)
    finally:
        await driver.close()

    # neo4j Records are immutable; materialise dicts before annotating with parsed timestamps.
    retired = [dict(r, _t=_parse(r["expired_at"])) for r in retired_rows]
    created = [dict(r, _t=_parse(r["created_at"])) for r in all_rows]
    retired = [r for r in retired if r["_t"] is not None]
    created = [c for c in created if c["_t"] is not None]

    reverted = sum(1 for r in retired if r.get("reverted"))
    print(f"edges total                       : {len(created)}")
    print(f"currently retired                 : {len(retired)}")
    print(f"  reverted by our own guard       : {reverted}")
    print()
    print(f"{'window':>8} {'decidable':>10} {'allow':>7} {'BLOCK':>7} {'block share':>12}")

    for window in WINDOWS_SECONDS:
        decidable = allow = block = 0
        for r in retired:
            cohort = [
                c for c in created
                if c["uuid"] != r["uuid"]
                and c["scope"] == r["scope"]
                and abs((c["_t"] - r["_t"]).total_seconds()) <= window
            ]
            if not cohort:
                continue
            decidable += 1
            shape = (r["src"], r["dst"], r["name"])
            if any(could_replace((c["src"], c["dst"], c["name"]), shape) for c in cohort):
                allow += 1
            else:
                block += 1
        share = f"{block / decidable:.1%}" if decidable else "n/a"
        print(f"{window:>7}s {decidable:>10} {allow:>7} {block:>7} {share:>12}")

    print(
        "\nBLOCK = no edge the originating write created could have replaced the retired one,\n"
        "so the retirement cannot be justified by anything that write actually asserted."
    )

    # --- the other direction: how much would the guard wrongly block? -------------------
    blocked_pairs = []
    decidable = allowed = 0
    for r in retired:
        cohort = [
            c for c in created
            if c["uuid"] != r["uuid"]
            and c["scope"] == r["scope"]
            and abs((c["_t"] - r["_t"]).total_seconds()) <= PRIMARY_WINDOW
        ]
        if not cohort:
            continue
        decidable += 1
        shape = (r["src"], r["dst"], r["name"])
        if any(could_replace((c["src"], c["dst"], c["name"]), shape) for c in cohort):
            allowed += 1
        else:
            blocked_pairs.append((r, cohort))

    rescued = []
    for r, cohort in blocked_pairs:
        name_shape = (_entity_key(r["src_name"]), _entity_key(r["dst_name"]), r["name"])
        match = next(
            (c for c in cohort
             if could_replace(
                 (_entity_key(c["src_name"]), _entity_key(c["dst_name"]), c["name"]), name_shape)),
            None,
        )
        if match is not None:
            rescued.append((r, match))

    n_blocked = len(blocked_pairs)
    if not n_blocked:
        return 0

    # Cohort-size stratification: bulk seed scripts write sequentially into one scope, so a large
    # cohort could manufacture the result. It does the opposite — more edges means more chances
    # something coincidentally looks like a replacement — so the small-cohort strata, where
    # attribution is unambiguous, are the ones that matter.
    print(f"\n--- block rate by cohort size at {PRIMARY_WINDOW}s (is attribution doing the work?) ---")
    print(f"{'cohort':>10} {'n':>6} {'blocked':>8} {'share':>8}   95% Wilson")
    sized = [(len(c), True) for _, c in blocked_pairs]
    for r in retired:
        cohort = [
            c for c in created
            if c["uuid"] != r["uuid"] and c["scope"] == r["scope"]
            and abs((c["_t"] - r["_t"]).total_seconds()) <= PRIMARY_WINDOW
        ]
        shape = (r["src"], r["dst"], r["name"])
        if cohort and any(could_replace((c["src"], c["dst"], c["name"]), shape) for c in cohort):
            sized.append((len(cohort), False))
    for lo, hi in [(1, 3), (4, 7), (8, 15), (16, 10**9)]:
        sub = [b for size, b in sized if lo <= size <= hi]
        if not sub:
            continue
        k, n = sum(sub), len(sub)
        ci_lo, ci_hi = _wilson(k, n)
        label = f"{lo}-{hi}" if hi < 10**9 else f"{lo}+"
        print(f"{label:>10} {n:>6} {k:>8} {k / n:>7.1%}   [{ci_lo:.1%}, {ci_hi:.1%}]")
    ci_lo, ci_hi = _wilson(n_blocked, decidable)
    print(f"{'all':>10} {decidable:>6} {n_blocked:>8} {n_blocked / decidable:>7.1%}   "
          f"[{ci_lo:.1%}, {ci_hi:.1%}]")

    still = n_blocked - len(rescued)
    print(f"\n--- entity-name relaxation ({allowed} allowed, {n_blocked} blocked) ---")
    print(f"  blocked AND unjustifiable by entity name : {still:>4}  ({still / decidable:.1%})")
    print(f"  blocked BUT rescued by entity name       : {len(rescued):>4}  "
          f"({len(rescued) / decidable:.1%})")
    print(
        "\nDO NOT read the rescued share as a wrong-block rate. It is computed with the SAME\n"
        "could_replace as the block decision, only fed entity names instead of uuids — one\n"
        "heuristic agreeing with a relaxed copy of itself, which notices nothing if the rule's\n"
        "idea of 'same relationship' is wrong. It detects exactly one failure mechanism\n"
        "(entity-resolution duplicates) and is blind to the rest.\n"
        "For an independent check see scripts/validate_invalidation_guard.py — which, as of\n"
        "2026-08-20, could not produce a usable number either: two judges from different model\n"
        "families reached only kappa 0.35 on whether a retirement was correct, across two rubrics\n"
        "and two judge pairings. The wrong-block rate is UNMEASURED, not small."
    )
    if rescued:
        print("\nrescued pairs — each is also an entity-resolution duplicate (names only):")
        for r, c in rescued[:8]:
            print(f"  retired  [{r['src_name']}] -{r['name']}-> [{r['dst_name']}]")
            print(f"  new      [{c['src_name']}] -{c['name']}-> [{c['dst_name']}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
