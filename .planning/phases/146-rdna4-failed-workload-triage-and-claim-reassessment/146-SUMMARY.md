---
phase: 146
title: RDNA4 failed workload triage and claim reassessment
status: completed
completed: 2026-06-08
---

# Phase 146 Summary

## Result

Phase 146 classified all 146 failed RDNA4 workload records across 35 problems.
The classification is captured in:

- `out/rdna4-failure-triage-v131/phase146-failure-triage.json`
- `out/rdna4-failure-triage-v131/phase146-failure-triage.md`

## Failure Classes

| Failure class | Count | Likely root cause | Next action |
|---|---:|---|---|
| `reference_gpu_oom` | 52 | Reference implementation failed with HIP OOM before candidate comparison. | Use larger VRAM or lower-memory reference path before claim upgrade. |
| `input_generation_gpu_oom` | 42 | Input generation failed with HIP OOM before benchmark execution. | Use larger VRAM or memory-efficient input generation; keep failed in denominator. |
| `execution_timeout` | 35 | CLI timed out after the configured 300 second timeout. | Increase timeout only for targeted reruns; keep failed for current denominator. |
| `gpu_oom_no_trace` | 12 | Shard failed with HIP OOM before emitting a per-workload trace. | Rerun only on larger VRAM or after memory reduction. |
| `user_function_gpu_oom` | 3 | Candidate/reference user function execution failed with HIP OOM. | Investigate kernel/workload memory footprint; rerun on larger VRAM if needed. |
| `timing_gpu_oom` | 1 | Timing collection failed with HIP OOM after functional execution path started. | Keep timing/failure boundary explicit; rerun timing on larger VRAM or reduced workload. |
| `incorrect_numerical` | 1 | Execution completed but failed correctness tolerance. | Debug numerical mismatch before any pass claim. |

## Claim Reassessment

No stronger RDNA4 claim is allowed from this evidence. Public docs now refer to
the v1.30/v1.31 evidence set: the bounded denominator remains unchanged, all
146 failed workloads remain visible, the 12 missing traces are classified as
`gpu_oom_no_trace`, and timing remains non-authoritative.

The allowed claim remains: bounded ready-subset RDNA4 `gfx1200` evidence for
the recorded host, commands, exclusions, and artifacts only. It is not full
235-problem paper validation, upstream SOLAR parity, NVIDIA B200 equivalence,
hosted leaderboard authority, CDNA3/MI300X validation, CDNA4 validation, or
broader AMD hardware validation.

