# Your temporal knowledge graph is quietly deleting true facts

*A defect in Graphiti's edge invalidation, and three attempts to measure it — two of which were wrong.*

**TL;DR** — On my production corpus, **70% of the facts Graphiti automatically retired were still
true** (28 of 40 hand-labelled cases, 95% CI [54.6%, 81.9%]). The failure is silent: no error, no
warning, the write reports success, and the fact simply stops appearing in search results. I found
it on 2026-07-27 and shipped a downstream guard. Then I measured the guard three times. The first
measurement was circular, the second showed my most recent addition to the guard changed nothing,
and the third showed the obvious improvement made it strictly worse. The guard that survives all
that is within *one case in forty* of simply switching invalidation off.

Reproduce: [`scripts/audit_invalidations.py`](../scripts/audit_invalidations.py),
[`scripts/validate_invalidation_guard.py`](../scripts/validate_invalidation_guard.py),
[`scripts/compare_guard_variants.py`](../scripts/compare_guard_variants.py).

---

## The system, in one paragraph

Synapse is a self-hosted temporal knowledge graph that AI coding agents across eleven of my projects
read from and write to over MCP. It runs on [Graphiti](https://github.com/getzep/graphiti), the
open-source engine behind Zep, over Neo4j. The premise of a *temporal* graph is that knowledge is
never deleted — it is superseded, with validity bounds, so "what did I believe in May, and when did
that change" stays answerable. That premise is the whole reason I chose this architecture.

## The symptom

Facts I knew I had written stopped coming back from `recall`. Not an error — an absence.

Digging: every fact edge carries `valid_at` / `invalid_at`. The missing ones had `invalid_at` set.
Something was retiring them. Nothing in my code retires anything, so it was happening inside
`add_episode`.

## The mechanism

Graphiti resolves contradictions on every write. Reading
`graphiti_core/utils/maintenance/edge_operations.py`, the invalidation-candidate search is:

```python
search(..., group_ids=[extracted_edge.group_id],
       config=EDGE_HYBRID_SEARCH_RRF, search_filter=SearchFilters())
```

`SearchFilters()` is empty. So the candidate set is *every edge in the group*, ranked by hybrid
semantic similarity, with **no requirement that a candidate share an entity with the new fact**.
That set goes to one LLM call which returns `contradicted_facts: list[int]` — a bare list of
indices, no justification, no confidence — and `resolve_edge_contradictions` commits `invalid_at` on
whatever comes back, gated only on temporal-overlap arithmetic.

Note what *is* scoped: `group_id`, so project isolation holds. Cross-project leakage isn't the bug.
The bug is that within a project, two facts that merely share vocabulary become candidates, and a
single index number from a language model is enough to retire one of them permanently.

## What it actually does to a corpus

The most common failure is a **narrower new fact retiring a broader true one**:

```
RETIRED:  <platform> uses Neo4j as one of its databases.
BY:       <service> stores its records in PostgreSQL, explicitly not Neo4j,
          because they are document-like entities.
```

`<service>` is one microservice inside `<platform>`. The platform still uses Neo4j — for a different
subsystem, which the very same write mentions. A statement about one service's storage choice retired
a true statement about the platform.

The second most common is **pure restatement**: a write that says the same thing in different words
retires the original.

```
RETIRED:  Billing is not yet implemented — the upgrade UI exists but no purchase
          flow is connected.
BY:       Billing has not yet been implemented — the upgrade UI exists but no
          purchase flow is connected.
```

Neither is a contradiction. (Examples are lightly generalised; the shapes are verbatim.)

---

## Measurement 1 — structural, and how far it got

If a write did not create any edge that *could* have replaced the retired one, the retirement had no
justification. "Could have replaced" has a precise structural meaning: same entity pair (either
direction — extraction direction isn't stable), or one endpoint in the same position plus the same
relation name.

Attribution matters, and this is where my first mistake lives. I joined retirements to writes on
`invalid_at`. That's *valid* time, which Graphiti backdates to when the fact stopped being true, so
it pairs each retirement with whatever happened to be written on an unrelated date. `expired_at` is
transaction time — when the write ran. Switching to it quadrupled the decidable set.

On 4,651 edges with 526 retired, of the 479 retirements whose originating write is identifiable:
**75.6% had no plausible replacement** (Wilson [71.5%, 79.2%]).

The obvious objection is that bulk seeding manufactured it — my corpus was partly built by scripts
that write sequentially into one scope, so the "cohort" could be an artifact. Stratifying by cohort
size answers it, and in the reassuring direction:

| cohort size | n | blocked | 95% Wilson |
|---|---|---|---|
| 2–3 (attribution unambiguous) | 31 | **100%** | [89.0%, 100%] |
| 4–7 | 27 | 81.5% | [63.3%, 91.8%] |
| 8–15 | 111 | 82.0% | [73.8%, 88.0%] |
| 16+ (seeding bursts) | 308 | 70.1% | [64.8%, 75.0%] |

Bigger cohorts *lower* the rate, because more edges mean more chances something coincidentally looks
like a replacement. The number is conservative.

## Measurement 2 — the one that was circular

The number above only bounds false positives. The question that decides whether anyone should change
a default is the other one: how often would this rule block a *real* supersession?

I tried to bound it cheaply. If extraction creates two nodes for one real entity, a genuine
supersession loses its shared uuid and gets blocked — so I re-ran the decision matching on entity
*name* instead of uuid. Only 8 of 362 flipped. **1.7%!** A 74% false-positive rate removed at under
2% cost. I wrote it up.

It's worthless. Both numbers come from the same function fed different identity keys — one heuristic
agreeing with a relaxed copy of itself. If the rule's notion of "same relationship" is wrong, both
move together and neither notices. And it detects exactly one failure mechanism while being blind to
every other.

I retracted it. The correct next step was the expensive one.

## Measurement 3 — labels

I sampled 40 attributed retirements, stratified 20/20 by what the structural rule decided, shuffled,
and judged each one blind to that decision: read the retired fact against the facts the attributed
write extracted, and consult the source repository when the text alone is ambiguous. That last part
matters — one case turned on whether a project's position cap was still 5%, which is in its README
and in no fact either judge was shown.

**28 of 40 were wrong retirements. 70%, Wilson [54.6%, 81.9%].**

The structural proxy's 75.6% sits inside that interval. Two methods sharing no mechanism, landing in
the same place. *That* is corroboration; measurement 2 was not.

I tried two LLM judges from different model families first, and they could not do it — Cohen's kappa
0.393, then 0.348 after sharpening the rubric with worked examples, upgrading the cloud judge, and
restricting to unambiguous cases. Two rounds of improving the instrument made agreement slightly
worse. The task needs context that a pair of extracted sentences doesn't carry.

---

## What a downstream guard can buy, and it isn't much

I have a guard that audits every write's own collateral damage and restores what it can't justify.
Scored against the same labels:

| rule | true facts silently lost | stale facts kept |
|---|---|---|
| unguarded (Graphiti as shipped) | **28** | 0 |
| structural test only | 13 | 5 |
| lexical test only | 0 | 11 |
| **my guard (structural AND lexical)** | **0** | 11 |
| invalidation disabled entirely | 0 | 12 |

These two errors are not interchangeable and I refuse to average them. Keeping a fact that should
have been retired is recoverable — it surfaces in a review queue. Silently losing a true one is not.
A single "accuracy" number would let a rule trade the expensive error for the cheap one and read as
an improvement.

Three things in that table are against me:

1. **The guard is within one case in forty of simply disabling invalidation** (11 stale vs 12). Its
   marginal value over turning the feature off is close to nothing.
2. **The structural test I added most recently changed no outcome.** Lexical-only is identical to the
   conjunction. I committed it with more confidence than it had earned.
3. **The obvious improvement made it worse.** Broadening the lexical test to accept named-entity
   substitutions — Suno → YouTube Audio Library, Kafka → Spring Modulith, the shape most
   supersessions in a software corpus actually take — introduced 3 silent losses and recovered 0
   stale facts.

## The conclusion I did not want

Once ~three quarters of retirements are unjustifiable, **no after-the-fact veto recovers a useful
signal.** It can only trade one error for the other. The fix has to be upstream, in the candidate
search, so the bad candidate never reaches the judge at all.

Which is exactly what [`getzep/graphiti#1729`](https://github.com/getzep/graphiti/pull/1729)
proposes — a structural guard on candidate selection. It has been open and unreviewed since
2026-08-04. The issue it closes, [#1728](https://github.com/getzep/graphiti/issues/1728), was filed
eight days after I shipped my own mitigation, by someone who hit the same thing on a different
backend with a different extraction model and hand-audited four cases, finding three wrong. Their
75% and my 70% are two independent corpora agreeing.

## What I'd tell anyone running a temporal knowledge graph

- **Audit `expired_at`, not `invalid_at`.** One is transaction time, the other is valid time and
  gets backdated. Confusing them will make any analysis you do roughly four times blinder.
- **Silent data loss is the failure mode to hunt first.** A crash tells you. This didn't.
- **A guard you haven't measured against labels is decoration.** Mine reverted 2% of retirements for
  three weeks and I believed it was working.
- **If your two measurements share a function, they are one measurement.**

## Limits

One corpus, one operator, 40 labels. The labels are mine, and I wrote the rule they judge — the only
thing arguing against self-serving bias is that they convicted it on both axes. Attribution uses a
±30s same-scope window, swept from 5s to 60s to show the result isn't an artifact of the choice. The
false-negative side is measured against my labels, not against a second labeller.
