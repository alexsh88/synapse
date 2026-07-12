# 4. Knowledge supersedes, it is never deleted

**Status:** Accepted — 2026-06-01

## Context

An agent's beliefs change. A project switches its testing convention; a decision
made "tentatively" in May becomes "locked" in July; a lesson learned the hard way
overrides an earlier assumption. A naive memory would overwrite or delete the stale
fact. That destroys the single most valuable question a long-lived memory can
answer: *what did I believe at time T, and why did it change?*

The alternatives:

1. **Mutate/delete in place.** Simple, but history is lost. "What did this project
   use in June vs. now" becomes unanswerable, and a bad write silently erases a good
   prior belief with no audit trail.
2. **Bi-temporal supersession.** Every fact carries validity bounds. Superseding a
   fact sets the old edge's `invalid_at` and stores the new one; the old fact stays
   in the graph, retrievable as-of any past time. This is native to Graphiti (see
   ADR-0001) and is Synapse's sacred rule.

## Decision

Adopt the temporal-supersede model. `forget` sets `invalid_at = now` (a temporal
end, not a `DELETE`). `update` invalidates the old fact and stores the new one.
Point-in-time retrieval (`recall(..., as_of=T)`, `brief`) filters on validity
bounds so a query returns what was believed at T. Nothing in Synapse's own code
issues a destructive delete on knowledge; curation may merge or archive, but never
loses a valid fact.

## Consequences

**Positive.** Full belief history, always. Contradictions are visible rather than
silently resolved. Curation and recall can reason about "current" vs. "historical"
knowledge with a single validity predicate.

**Negative — and these were real bugs, not hypotheticals.** Getting supersession
correct is subtle, and several early implementations were quietly wrong:

- `update` was *forget-then-remember*. If the remember failed, the old fact was
  already invalidated — knowledge silently vanished from recall while the call
  returned `success: True`. Fixed to store-new-first, invalidate-only-on-success.
- `forget` overwrote `invalid_at` unconditionally, so re-forgetting an
  already-superseded fact *moved* its supersession timestamp and corrupted the "what
  did I think in May" answer. Fixed to `coalesce(invalid_at, now)`.
- The "live fact" Cypher predicate was hand-written across six-plus modules and
  drifted — the `coalesce` inconsistency above was a direct symptom of that
  duplication.

The lesson: a temporal model is only as trustworthy as its least-careful mutation
path. We now assert on the emitted Cypher in invariant tests (a concurrency test for
transactional safety, an end-to-end supersede test for history) so these
correctness properties are guarded, not just asserted in a design doc. The honest
cost accepted: the graph grows monotonically (superseded facts are never reclaimed),
and every read must filter validity — complexity we pay on every query in exchange
for an answer we can never reconstruct if we throw it away.
