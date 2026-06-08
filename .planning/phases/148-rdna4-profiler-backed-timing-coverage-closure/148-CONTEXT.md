# Phase 148 Context: RDNA4 Profiler-Backed Timing Coverage Closure

## Goal

Build the denominator-aware coverage machinery needed to pursue complete
profiler-backed `rocprofv3` timing coverage across the expected 235 SOL
ExecBench problem denominator.

## Starting Point

- v1.30/v1.31 produced bounded RDNA4 ready-subset execution evidence.
- Existing 121 v1.31 timing sidecars are PyTorch/device-event fallback, not
  profiler-backed kernel activity timing.
- A quick task fixed `eval_driver.py` graceful profiler finalization and proved
  a real RDNA4 Triton smoke can produce `_kernel_trace.csv`.

## Constraints

- Full expanded profiler timing execution is long-running and hardware-bound;
  the first phase slice should create auditable coverage reporting and avoid
  claiming coverage that is not backed by `rocprofv3`.
- The full 235-problem denominator must remain visible even when some problems
  are readiness-blocked or missing timing.
- Existing user edits in unrelated docs must not be modified.
