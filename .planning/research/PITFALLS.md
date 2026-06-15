# Research: Pitfalls for v1.36 SOL Agent Feedback Sidecar Producer

**Date:** 2026-06-15

## Pitfalls

- **Trace schema drift:** adding fields to canonical Trace JSONL would break the
  compatibility boundary HIP expects. Keep feedback in sidecars only.
- **Authority leakage:** a sidecar that can claim score, timing, evidence-tier,
  paper-parity, leaderboard, or cutover authority would undermine existing
  release guardrails. Use literal false authority fields and validation tests.
- **Prompt leakage:** raw profiler dumps, trace rows, source code, stderr dumps,
  or absolute temp paths are unsuitable for agent prompts. Emit compact cited
  summaries only.
- **Stale feedback:** reused trace paths or resumed runs can make an old sidecar
  look current. Include identity fields and make stale checks explicit.
- **Ad hoc bottleneck labels:** HIP has a closed `ProfileDigest` taxonomy. SOL
  should emit stable categories plus unknown/limitation fallbacks rather than
  unbounded free-form labels.
- **Optional-tool confusion:** profiler or static evidence may be unavailable.
  Sidecar absence or unavailable status must never fail evaluation.

## Prevention Strategy

- Model authority boundaries as strict schema fields.
- Keep capability tokens optional and backward compatible.
- Add fixtures for valid, missing, unavailable, malformed, stale, and
  contradictory-authority cases.
- Reuse existing checksum and artifact-reference helpers instead of inventing a
  parallel identity model.
