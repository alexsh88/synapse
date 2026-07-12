# Building Synapse: an engineering narrative

## What I was actually trying to solve

I run a lot of projects — ten of them — and I work on them with AI coding agents.
Every session, the agent started from zero. It re-derived the same conventions,
re-asked the same architectural questions, re-learned the same lessons I'd already
paid for once. The knowledge existed; it just lived in my head, in scattered
markdown, in commit messages nobody re-reads. Nothing compounded.

Synapse is my answer: a single temporal knowledge graph that every agent, across
every project, reads from at session start and writes back to as it works. A brain
that remembers *for* me, so the twentieth session on a project starts where the
nineteenth left off. One tool call — `brief(project_id)` — loads the context. The
whole system exists to make that one call worth making.

## The five hardest trade-offs

**Supersede, never delete.** The tempting design is to overwrite stale facts. I
refused, because the question I most wanted answered was "what did I believe in May,
and why did it change?" — and a `DELETE` throws that away forever. So every fact
carries validity bounds and superseding sets `invalid_at` rather than removing
anything. The cost is real: the graph only grows, and every read has to filter
validity. I paid it on purpose.

**Local-first embeddings, and the cost math that surprised me.** I assumed cost
would decide the embedder. It didn't — my corpus is thousands of facts, so even the
priciest hosted API costs cents a year. What actually decided it was owning the data
and avoiding a one-way door: the vector dimension locks at first ingestion (1024,
for BGE-M3), and changing it later means re-embedding everything. Local won on
privacy and lock-in, with cost a wash. Running it on my own GPU was free per token
and faster than a cloud round-trip.

**Extraction routing when the credits ran out.** Mid-build, my Anthropic credits hit
zero — and writes stopped, because extraction was the one paid call in the stack.
That forced the design I should have built anyway: a credit-aware fallback that flips
cloud→local automatically, with the cooldown shared across all ten MCP processes via
Redis so they don't each independently rediscover that credits are gone. Sonnet stays
the default because quality is the point; local is the resilience lever.

**Retrieval scoring, and the reranker that lied.** Retrieval quality *is* the value —
if `recall` returns junk, the whole system is worthless. While tuning, I found the
ranking signal was a linear ramp over list position, not similarity at all. Digging
deeper: Graphiti's reranker returns *rank positions*, not similarity scores. I'd been
weighting positions as if they were cosines. The fix was to compute real cosine per
result and add a `min_relevance = 0.72` floor to kill confident junk — BGE-M3 scores
off-topic facts around 0.65 and on-topic ones at 0.80+, so 0.72 is the calibrated
line between them. That one floor took retrieval violations from 3 to 0.

**Curation safety.** Background curation merges duplicates and archives decayed
knowledge — exactly the operations that can silently destroy a valid fact. I built a
`verify_no_loss` check and, embarrassingly, left it *disconnected from every mutation
path* for a while. Wiring it in — so a merge that would lose knowledge raises instead
of proceeding, and backups scope to the affected neighborhood — was the difference
between a safety net and a decoration.

## What broke, and what it taught me

**BGE-M3 returned NaN.** Writes failed intermittently with `Ollama 500: "json:
unsupported value: NaN"`. It's a known Ollama flash-attention instability, input-
sensitive, so the *same* borderline string would pass or fail depending on GPU state
— which is why my first "it's not the content" diagnosis was wrong. `OLLAMA_FLASH_
ATTENTION=0` fixed it (a string that failed 15/15 went to 0/20). Lesson: when a bug
is non-deterministic, distrust every conclusion you drew from a single run.

**The 14% silent-empty tail.** When I A/B'd local extraction, the danger wasn't
inaccuracy — it was malformed JSON that broke the parser and extracted *nothing*,
returning a clean success while the facts vanished. Worse, the failures clustered on
the richest episodes. Strict `json_schema` decoding cut it dramatically and nearly
doubled edge capture, but a ~14% tail remained. Silent data loss is the failure mode
I now hunt for first.

**The fail-open triage bug.** The write pipeline triages whether a fact is worth
storing. On a JSON parse failure it defaulted to `worth_storing = True` — meaning the
noise filter switched *off* exactly when the weak local model was producing the
malformed output. Fail-open in a filter is fail-loud in the graph. I made it fail
closed.

## How I know it works

Not because it feels done — because it's measured. There are 201 tests, including
invariant tests that assert on the *emitted Cypher* for the temporal guarantees
(concurrency safety, end-to-end supersession) rather than trusting a design doc.
Retrieval has an eval harness with negative cases (including a check that one
project's knowledge never leaks into a sibling with a similar name) and a regression
gate that fails CI if MRR drops more than 5% or any violation reappears. The tuning
story is in the numbers: MRR 0.484 → 0.531 with no hits lost, violations 3 → 0. The
eval also earns its keep by telling me when a miss is a *curation* gap (a fact
invalidated with no successor) rather than a ranking bug — a distinction I couldn't
make before.

## What I'd do differently, and at 10× scale

I'd centralize the "live fact" predicate from day one. It was hand-written across six
modules and drifted, and the drift *was* one of the supersession bugs — duplicated
truth is duplicated bugs. I'd also build the eval harness before the ranking code,
not alongside it; I spent effort I couldn't measure.

At 10× the corpus, the time-bombs are known and marked: per-write dedup and curation
pair-scans that do full or O(n²) cosine scans instead of using the vector index. At
100× I'd revisit the Qdrant decision (deliberately reversible, with documented
triggers), split the `KnowledgeEngine` god-object, and put real auth in front of the
MCP interface. But those are scale problems, and I built for the scale I have — a
single deep brain, not a fleet — on purpose.
