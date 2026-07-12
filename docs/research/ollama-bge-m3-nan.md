# Research — Ollama bge-m3 NaN embeddings (and the fix)

**Date:** 2026-05-31
**Problem:** bge-m3 on Ollama intermittently returned NaN embeddings →
`Ollama 500: "json: unsupported value: NaN"`, failing Synapse writes during Phase 4.
**Resolution:** ✅ **`OLLAMA_FLASH_ATTENTION=0`** — validated (15/15-failing string → 0/20).

## What it is

A **known, well-documented Ollama bug**, not a Synapse defect:

- bge-m3 (and other embedders — nomic-embed-text, qwen3-embedding) produce **NaN values
  for certain inputs**. Go's `encoding/json` can't serialize NaN, so Ollama returns a
  cryptic **500 "json: unsupported value: NaN"**. The crash is a *secondary symptom*.
- **Root cause is the flash-attention computation** — disabling it prevents the NaN at
  the source (maintainers' own note in the issue thread).
- It's **input-sensitive**: e.g. a specific text NaNs, and removing/replacing the last
  word fixes it; one report saw 76/1217 texts fail with no obvious pattern. Suspected
  fp16 numerical instability. This explains the "on-the-edge" intermittency we saw
  (the same borderline string flips between OK and NaN depending on GPU/attention state).

Tracking issues: [ollama#13572](https://github.com/ollama/ollama/issues/13572),
[#14657](https://github.com/ollama/ollama/issues/14657),
[#9639](https://github.com/ollama/ollama/issues/9639),
[#12921](https://github.com/ollama/ollama/issues/12921).

## Correcting our earlier diagnosis

During debugging we *thought* we'd ruled out content (the failing string embedded fine in
isolation — sometimes). The research shows it's **both** input-sensitivity **and**
flash-attention instability: borderline inputs tip into NaN non-deterministically. So
"content" wasn't fully ruled out — it's content-on-the-numerical-edge.

## Fix (applied)

1. **`OLLAMA_FLASH_ATTENTION=0`** on the Ollama server, then restart. ← the real fix.
   Persisted with `setx` so it survives restarts. Validated: the string that failed
   15/15 → **0/20**; full 7-tool live smoke → **0 embed failures**.
2. **Update Ollama** to a build including [PR #14739](https://github.com/ollama/ollama/pull/14739),
   which validates embeddings and returns a *clear* error instead of crashing on NaN
   (belt-and-suspenders; our retry/fallback then handles it).
3. Defense-in-depth in code: `RetryingOpenAIEmbedder` (retry + per-item batch fallback),
   concurrency cap. Fallback model if ever needed: `nomic-embed-text` (768-dim).

## Operator durability checklist

- [x] `OLLAMA_FLASH_ATTENTION=0` set in the user environment (`setx`).
- [ ] Confirm the auto-started Ollama app/service inherits it after a reboot (set it in
      Ollama's settings or system env if the desktop app doesn't pick up the user var).
- [ ] Update Ollama to latest (graceful NaN handling).

## Sources
- [ollama#13572 — bge-m3 NaN, no detectable pattern](https://github.com/ollama/ollama/issues/13572) ·
  [#14657 — bge-m3 NaN on specific docs](https://github.com/ollama/ollama/issues/14657) ·
  [#9639 — unsupported value NaN](https://github.com/ollama/ollama/issues/9639) ·
  [#12921 — qwen3 fp16 NaN](https://github.com/ollama/ollama/issues/12921)
- [bge-m3 NaN write-up (StepCodex)](https://www.stepcodex.com/en/issue/bge-m3-only-returns-nan-on) ·
  [LightRAG #1870 (same error downstream)](https://github.com/HKUDS/LightRAG/issues/1870) ·
  [Ollama flash attention guide](https://craftrigs.com/guides/ollama-flash-attention-enable-gpu-benchmark/)
