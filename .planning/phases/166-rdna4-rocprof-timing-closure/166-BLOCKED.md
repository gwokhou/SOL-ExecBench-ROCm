---
phase: 166
status: blocked
blocked_at: "2026-06-08"
resolved_at: "2026-06-09"
resolution: "User accepted current-device profiler-closure OOM blocker class."
---

# Phase 166 Blocked: RDNA4 Rocprof Timing Closure

## Result

Phase 166 is blocked on current-device RDNA4 memory/workload behavior. The
host has visible RDNA4 `gfx1200` hardware and `rocprofv3`, and escalated
PyTorch ROCm can access the GPU, but bounded profiler closure attempts did not
produce complete profiler-backed timing for the tested targets.

## Preflight

- `which rocprofv3`: `/usr/bin/rocprofv3`
- `rocminfo`: reports AMD Radeon Graphics `gfx1200`
- `rocm-smi`: reports one GPU in `auto` performance level
- sandboxed `uv run` PyTorch: HIP 7.1.25424, `torch.cuda.is_available() == False`
- escalated `uv run` PyTorch: HIP 7.1.25424, one AMD Radeon Graphics device

The GPU is usable only with escalated execution from this environment.

## Attempt 1: Full-Problem Smoke

Command output root:

- `out/rdna4-rocprof-timing-closure-20260608-smoke/`

Target:

- `L1/028_hybrid_attention_mask_preparation`

Outcome:

- `selected_targets`: 1
- `succeeded`: 0
- `profiler_blocked`: 1
- `trace_status_counts`: `{"PASSED": 2}`
- `rocprofv3` wrote kernel trace CSV rows, but the process exited 1.
- Failure: HIP OOM during correctness error-stat computation after two passing
  traces.

## Attempt 2: Workload-Sharded Existing Profiler-Blocked Target

Command output root:

- `out/rdna4-rocprof-timing-closure-20260608-l1028-sharded/`

Target:

- `L1/028_hybrid_attention_mask_preparation`

Outcome:

- `partial_profiler_backed`: 1
- `profiled_workload_count`: 6
- `expected_workload_count`: 12
- `trace_status_counts`: `{"PASSED": 6, "RUNTIME_ERROR": 3}`
- Some workload slices still failed or were profiler-blocked.
- This target remains partial and cannot be promoted to `profiler_backed`.

## Attempt 3: Workload-Sharded Timing-Fallback Target

Command output root:

- `out/rdna4-rocprof-timing-closure-20260608-l1053-sharded/`

Target:

- `L1/053_gaussian_topk_sparse_activation`

Outcome:

- `partial_profiler_backed`: 1
- `profiled_workload_count`: 11
- `expected_workload_count`: 12
- `trace_status_counts`: `{"PASSED": 11}`
- Offset 9 remained `profiler_blocked`.

## Attempt 4: Single-Slice Retry

Command output root:

- `out/rdna4-rocprof-timing-closure-20260608-l1053-offset9-retry/`

Target:

- `L1/053_gaussian_topk_sparse_activation`, workload offset 9

Outcome:

- `profiler_blocked`: 1
- Failure: HIP OOM during correctness error-stat computation:
  `torch.OutOfMemoryError: HIP out of memory. Tried to allocate 4.00 GiB.`

## Blocker

The current 16GB RDNA4 host cannot close at least one actual timing-fallback
target under workload-sharded profiler replacement because a single workload
still OOMs in the correctness path. Continuing across all 46 timing-fallback
targets would spend substantial GPU time while known current-device OOM
blockers remain unresolved.

## Required Decision

Choose one before continuing Phase 166:

- Accept a new current-device blocker class for profiler closure OOMs in
  correctness/error-stat paths and keep those targets out of the eligible
  `profiler_backed` numerator.
- Provide a larger-memory AMD GPU for the remaining fallback/profiler-blocked
  targets.
- Explicitly authorize a long-running, best-effort workload-sharded sweep over
  the remaining fallback targets, accepting that many may become partial or
  OOM-blocked rather than fully closed.

## Claim Boundary

No RDNA4 timing claim is upgraded by these attempts. Phase 166 did not produce
complete profiler-backed timing closure for the tested targets, and Phase 167
clock-lock evidence plus Phase 168 release bundle should not proceed until this
blocker is resolved or explicitly accepted.
