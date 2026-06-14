---
status: complete
quick_id: 260613-prof
slug: try-fix-rdna4-l2-profiler-blockers
description: Try to fix 24 RDNA4 L2 profiler-blocked targets that are not memory-cap blocked
created_at: 2026-06-13T16:45:00+08:00
---

# Quick Task 260613-prof: Try Fix RDNA4 L2 Profiler Blockers

## Goal

Attempt to turn the 24 non-memory-cap `L2/*` profiler-blocked targets into
profiler-backed timing on this host.

## Scope

In scope:

- 23 targets with `rocprofv3 command failed with exit code -11`.
- 1 target with `rocprofv3 did not produce kernel activity rows`.

Out of scope:

- 5 MoE targets blocked by dynamic timing-input memory preflight. Their
  estimated timing input peaks are 24.13-48.38 GiB, which is not realistic for
  the current 15.9 GiB VRAM host without changing benchmark/timing semantics.

## Baseline

- Final prior coverage: `out/rdna4-ready-missing-profiler-closure-20260613/coverage-stage2/`
- `profiler_backed_problems`: 121 / 235
- `ready_missing_profiler_timing_problems`: 0
- `profiler_blocked_problems`: 35

## Strategy

1. Reproduce the single no-kernel-row target (`L2/018`) with focused profiler
   output and inspect parser/CSV assumptions.
2. Reproduce one representative `rocprofv3 exit -11` target with the smallest
   workload slice.
3. Try profiler invocation or isolation changes that keep benchmark semantics
   intact.
4. If a fix works, batch the 24 target set and recompute coverage.

## Claim Boundary

This quick attempts to improve profiler-backed timing coverage only for targets
that are not currently blocked by memory-cap preflight. It must not relax memory
safety or silently count fallback timing as profiler-backed evidence.
