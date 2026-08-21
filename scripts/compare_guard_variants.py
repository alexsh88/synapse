"""Score every invalidation-guard variant against hand labels, by ERROR TYPE.

    python -m scripts.compare_guard_variants --labels invalidation-labels.md

Takes a labelling file produced by ``scripts/validate_invalidation_guard.py --export`` and filled
in by a human, plus the ``.key.json`` written beside it, and reports what each candidate rule would
have done: doing nothing (Graphiti unmodified), the structural test alone, the lexical test alone,
their conjunction (what ships), their disjunction, and disabling invalidation entirely.

The two errors are not interchangeable, and the report refuses to average them. Under R8, keeping a
fact that should have been retired is recoverable and visible in the review queue; silently losing a
true one is not. A single "accuracy" number would let a rule trade the expensive error for the cheap
one and look like an improvement, which is exactly the mistake this script exists to prevent.

No network, no database — it reads the labelling file and re-derives each rule offline.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

from synapse.core.consolidation_engine import invalidation_is_credible, subject_overlap

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_CASE_RE = re.compile(r"^##\s+(\d+)\s*$")
_ANSWER_RE = re.compile(r"^ANSWER:\s*([01])\s*$")


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def parse_cases(path: Path) -> dict[int, dict[str, Any]]:
    """Pull (old fact, new facts, label) out of the labelling file."""
    cases: dict[int, dict[str, Any]] = {}
    current: int | None = None
    mode: str | None = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.rstrip()
        m = _CASE_RE.match(line)
        if m:
            current = int(m.group(1))
            cases[current] = {"old": "", "new": [], "label": None}
            mode = None
            continue
        if current is None:
            continue
        if line.startswith("OLD FACT"):
            mode = "old"
        elif line.startswith("NEW FACTS"):
            mode = "new"
        elif (a := _ANSWER_RE.match(line)) is not None:
            cases[current]["label"] = int(a.group(1))
            mode = None
        elif mode == "old" and line.strip():
            cases[current]["old"] += line.strip() + " "
        elif mode == "new" and line.strip().startswith("- "):
            cases[current]["new"].append(line.strip()[2:])
    return cases


def report(rows: list[dict[str, Any]], rule: str, name: str) -> None:
    """`rule` names a boolean field: True = uphold the retirement, False = revert it."""
    lost = sum(1 for r in rows if r[rule] and r["label"] == 0)      # upheld a wrong retirement
    stale = sum(1 for r in rows if not r[rule] and r["label"] == 1)  # reverted a correct one
    n = len(rows)
    lo_l, hi_l = wilson(lost, n)
    lo_s, hi_s = wilson(stale, n)
    print(f"{name:<38}{lost:>5} [{lo_l:>5.1%},{hi_l:>6.1%}]{stale:>7} [{lo_s:>5.1%},{hi_s:>6.1%}]")


def main() -> int:
    p = argparse.ArgumentParser(description="Compare invalidation-guard variants against labels")
    p.add_argument("--labels", type=Path, default=Path("invalidation-labels.md"))
    args = p.parse_args()

    key_path = Path(str(args.labels) + ".key.json")
    if not args.labels.exists() or not key_path.exists():
        print(f"[error] need {args.labels} and {key_path} "
              "(produced by validate_invalidation_guard.py --export).")
        return 2

    cases = parse_cases(args.labels)
    key = json.loads(key_path.read_text(encoding="utf-8"))["cases"]

    rows: list[dict[str, Any]] = []
    for idx, c in sorted(cases.items()):
        if c["label"] is None:
            continue
        old = c["old"].strip()
        structural = bool(key[str(idx)]["structural_allows"])
        lexical = any(invalidation_is_credible(old, n) for n in c["new"])
        rows.append({
            "idx": idx,
            "label": c["label"],
            "structural": structural,
            "lexical": lexical,
            "shipped": structural and lexical,
            "either": structural or lexical,
            "unguarded": True,
            "disabled": False,
            "overlap": max((subject_overlap(old, n) for n in c["new"]), default=0.0),
        })

    if not rows:
        print("[error] no answers found — fill in the ANSWER: lines first.")
        return 1

    wrong = sum(1 for r in rows if r["label"] == 0)
    print(f"{len(rows)} labelled retirements: {wrong} wrong, {len(rows) - wrong} correct\n")
    print(f"{'rule':<38}{'SILENT LOSSES':>25}{'STALE KEPT':>26}")
    for rule, name in [
        ("unguarded", "unguarded (Graphiti as shipped)"),
        ("structural", "structural test only"),
        ("lexical", "lexical test only"),
        ("either", "structural OR lexical"),
        ("shipped", "SHIPPED (structural AND lexical)"),
        ("disabled", "invalidation disabled entirely"),
    ]:
        report(rows, rule, name)

    print("\nSilent losses are the expensive error (R8): a true fact gone, with no error and no")
    print("warning. Stale kept is the cheap one — the contradiction review queue can still catch it.")

    bad = [r for r in rows if r["shipped"] != (r["label"] == 1)]
    if bad:
        print(f"\nwhere the shipped guard disagrees with the labels ({len(bad)} cases):")
        for r in bad:
            kind = "upheld a WRONG retirement" if r["shipped"] else "reverted a CORRECT one"
            print(f"  case {r['idx']:>3}  {kind:<26} structural={str(r['structural']):<5} "
                  f"lexical={str(r['lexical']):<5} best_overlap={r['overlap']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
