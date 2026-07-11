# Task 1 Report: Calibration Contract and Profiler Environment

## Delivered

- Added strict `sol_execbench.hardware_calibration.v1` artifact and candidate
  contracts with the three allowed evidence states.
- Added conservative best-three sampling, including finite/positive, seven
  sample minimum, and 5% retained-spread validation.
- Added lazy, file-locked ROCm Compute Profiler virtual-environment management
  scoped to `.artifacts/rocprof-compute`, with dependency provenance manifests.
- Added isolated `rocprof-compute profile --bench-only` invocation that sets
  `VIRTUAL_ENV`, prepends managed `bin` to `PATH`, and sets
  `PYTHONNOUSERSITE=1`.
- Added injected filesystem/subprocess seams and offline/no-auto-install
  outcomes that remain `unknown` without ambient-package fallback.

## Validation

`uv run pytest tests/sol_execbench/core/scoring/hardware_calibration/test_models.py tests/sol_execbench/core/scoring/hardware_calibration/test_statistics.py tests/sol_execbench/core/scoring/hardware_calibration/test_rocprof_compute.py -n 0 -v`

Result: 4 passed.

`uv run --with ruff ruff check src/sol_execbench/core/scoring/hardware_calibration tests/sol_execbench/core/scoring/hardware_calibration`

Result: passed.

## Scope review

No CLI command, HIP probe, hardware-model v3, or score authority gate was
added. The managed environment does not install unless `auto_install=True` is
explicitly supplied by a future calibration caller.

## Review Fixes (2026-07-10)

- Tightened versioned JSON parsing so schema/status/timestamp and candidate
  string fields reject non-string values, and values/samples reject numeric
  strings, booleans, non-finite values, and other non-JSON numeric types with
  controlled `ValueError`s.
- Added `parse_roofline_metrics()`, which retains a SHA-256 of raw profiler CSV
  output and emits only `unknown` candidates for missing, unrecognised, or
  insufficiently evidenced metrics; it never manufactures calibration values.
- Reused profiler environments now verify that the recorded system launcher
  exists and is executable before returning `measured` availability.

### Focused test output

```text
============================== 14 passed in 0.58s ==============================
```

The focused command was:

```text
uv run pytest tests/sol_execbench/core/scoring/hardware_calibration/test_models.py tests/sol_execbench/core/scoring/hardware_calibration/test_statistics.py tests/sol_execbench/core/scoring/hardware_calibration/test_rocprof_compute.py -n 0 -v
```

## Second Review Fixes (2026-07-10)

- Centralized launcher existence/executability checks before any managed
  environment installation and in the sole final `measured` construction path.
  This covers reused, newly created, and lock-race-reused environments.
- Made installed-distribution manifest parsing strict: only a JSON list of
  strings is accepted. Missing manifests remain `manifest_unavailable`; valid
  JSON with the wrong shape returns `unknown` with
  `rocprof_compute_manifest_invalid`.
- Added focused tests for missing launcher before `uv`, a launcher disappearing
  after a lock-race recheck, and both malformed manifest shapes.

### Focused test output

```text
============================== 17 passed in 0.60s ==============================
```

```text
uv run pytest tests/sol_execbench/core/scoring/hardware_calibration/test_models.py tests/sol_execbench/core/scoring/hardware_calibration/test_statistics.py tests/sol_execbench/core/scoring/hardware_calibration/test_rocprof_compute.py -n 0 -v
```
