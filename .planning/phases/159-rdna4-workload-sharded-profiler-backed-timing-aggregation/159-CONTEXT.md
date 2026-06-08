---
status: planned
---

# Phase 159 Context

## Problem

Full-problem `rocprofv3` replacement attempts can OOM or exceed practical
timeouts on large RDNA4 problems. Raising process memory caps and timeouts is a
useful circuit breaker, but it does not solve the structural problem: a single
profiler session currently owns too much validation and trace-collection work.

`L1/037_flux_feedforward_gelu_approximate` demonstrated the shape of the issue.
Offsets 0-15 all pass as independent profiler-backed workload slices, but the
full problem times out because the aggregate profiler lifecycle is too long.

## Decision

Phase 159 will turn workload slicing from a diagnostic-only escape hatch into a
first-class, auditable profiler timing path:

- run each expected workload independently under `rocprofv3`;
- record deterministic per-workload manifest entries;
- aggregate only complete, passed, profiler-backed workload evidence into a
  problem-level `profiler_backed` timing sidecar;
- keep partial, timeout, OOM, profiler-crash, and readiness failures visible
  without counting them as full coverage.

## Claim Boundary

Per-workload evidence may satisfy problem-level profiler-backed coverage only
after aggregation proves that every expected workload for the problem is
present, passed, profiler-backed, and trace-backed. Missing, failed, fallback,
or diagnostic-only slices must remain partial evidence.
