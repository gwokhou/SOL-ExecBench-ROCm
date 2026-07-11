# Task 2 report — Architecture adapters and HIP/ROCm orchestration

## Delivered

- Added runtime GPU discovery with injectable `GpuRuntime` support and adapters for
  `gfx12*`, `gfx94*`, and `gfx95*`.
- Declared portable FP32 vector and streaming-copy profiles, plus family-specific
  paths, for every supported architecture family.
- Added an injectable HIP probe lifecycle that only emits `measured` after compile,
  execution, numerical correctness, and stability all pass. Explicit unsupported
  results are `unavailable`; missing, failed, and invalid probes are `unknown`.
- Added calibration orchestration, including RDNA4 `lock_clocks()`/`unlock_clocks()`
  lifecycle reuse, provisional diagnostics when clock locking is absent or fails,
  and rejection when clock locking is required.
- Optional rocprof-compute bench-only output is retained as checksum-backed profile
  metadata; recognised metrics are mapped to declared matrix keys without replacing
  HIP measurements.

## Verification

`uv run pytest tests/sol_execbench/core/scoring/hardware_calibration/test_environment.py tests/sol_execbench/core/scoring/hardware_calibration/test_hip_probe.py tests/sol_execbench/core/scoring/hardware_calibration/test_builder.py -n 0 -v` — 7 passed.

`uv run --with ruff ruff check …` — clean for Task 2 source and tests.

## Concern

The HIP probe deliberately exposes injected compile/run/check seams. A future CLI
task must provide a concrete HIP compilation/execution implementation; this task
does not perform real HIP compilation in unit tests.

## Review follow-up (2026-07-10)

- Added `HipCommandBackend` and `default_hip_probe()`. The backend writes a
  self-checking generated HIP source, invokes `hipcc`, executes the binary, parses
  seven timing samples, and treats non-zero exit, malformed output, and absent or
  failed `hipcc` as explicit `unknown` evidence. Dependency injection remains
  available for unit tests.
- `CalibrationRequest` now obtains that default backend when no probe is injected.
- Profiler recognition now parses only observed CSV metric labels, maps each label
  to every matching matrix key, and records raw-output SHA-256. Empty/missing or
  unrecognised CSV is `unknown`, not collected.
- Added tests for runtime discovery, RDNA4 lock/unlock lifecycle, malformed output,
  default backend failure/default selection, repeated profiler attribution, and
  empty profiler output.

Exact verification evidence:

```text
$ uv run pytest tests/sol_execbench/core/scoring/hardware_calibration/test_environment.py tests/sol_execbench/core/scoring/hardware_calibration/test_hip_probe.py tests/sol_execbench/core/scoring/hardware_calibration/test_builder.py -n 0 -v
============================== 14 passed in 0.59s ==============================
$ uv run --with ruff ruff check src/sol_execbench/core/scoring/hardware_calibration/environment.py src/sol_execbench/core/scoring/hardware_calibration/hip_probe.py src/sol_execbench/core/scoring/hardware_calibration/builder.py tests/sol_execbench/core/scoring/hardware_calibration/test_environment.py tests/sol_execbench/core/scoring/hardware_calibration/test_hip_probe.py tests/sol_execbench/core/scoring/hardware_calibration/test_builder.py
All checks passed!
```
