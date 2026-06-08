---
status: planned
---

# Phase 161 Context

## Problem

The remaining `partial_profiler_backed` bucket mixes several different
conditions: invalid references, runtime errors, profiler evidence gaps, and
partial workload success. Retrying profiler collection cannot close targets
whose blocker is correctness or runtime behavior.

## Decision

Phase 161 will create a deterministic failure classification ledger for partial
profiler targets. It will classify observed failure modes and assign a closure
decision before any further profiler work is attempted.

## Focus Target

`L1/076_batched_expert_forward` is the clearest deep-dive target because the
current full attempt reports `RUNTIME_ERROR` for all 14 workloads.
