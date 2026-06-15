# Research Summary: v1.36 SOL Agent Feedback Sidecar Producer

**Date:** 2026-06-15

## Summary

HIP Playground v1.26 needs SOL-ExecBench-ROCm to produce an optional
diagnostic-only feedback/profile summary sidecar that can guide hip-agent's next
experiment. SOL already has the right implementation patterns: strict Pydantic
sidecar models, `contract --json` capability metadata, diagnostic-only authority
flags, checksum helpers, artifact refs, evaluation-stability summaries, and
guardrail tests.

## Stack Additions

- Add a small SOL-owned agent-feedback schema/generator module.
- Extend evaluator contract optional capabilities.
- Add CLI persistence beside canonical trace output.
- Add documentation and fixtures for HIP consumer tests.

## Table Stakes

- `trace.jsonl.agent-feedback.json` is optional and diagnostic-only.
- Canonical trace remains the source of correctness, timing, scoring, and status
  truth.
- Feedback includes bounded bottlenecks, recommendations, limitations,
  freshness identity, and compact artifact citations.
- Invalid or stale sidecars are represented as diagnostic states, never as
  evaluation failures.

## Watch Outs

- Do not put feedback fields into canonical Trace JSONL.
- Do not expose raw profiler dumps, raw trace rows, full source, or absolute
  temp paths to prompt-facing summaries.
- Do not let sidecar content promote evidence tier, score authority, confirmed
  improvement, release gates, cutover eligibility, paper parity, or leaderboard
  readiness.
