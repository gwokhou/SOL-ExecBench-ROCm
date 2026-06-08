---
quick_id: 260608-kjo
slug: rdna4-profiler-backed-timing-smoke-slice
status: complete
completed_at: 2026-06-08
---

# Quick Task 260608-kjo Summary

## Completed

- Added `scripts/run_rdna4_profiler_timing_smoke.py`, a bounded RDNA4 timing
  smoke entry point for the Triton RMSNorm example.
- The script stages a limited workload, routes `languages=["triton"]` through
  `collect_source_timing_evidence()`, writes `timing.json`, `summary.json`, and
  `summary.md`, and exits nonzero unless profiler-backed `rocprofv3` timing is
  collected or `--allow-fallback` is explicitly set.
- Added CPU-safe tests that fake `rocprofv3` CSV output and verify
  `profiler_collected=true`, kernel-activity metadata, workload limiting, and
  fallback failure behavior.

## Verification

- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_rdna4_profiler_timing_smoke.py -q`
  - `3 passed in 4.28s`
- `UV_CACHE_DIR=/tmp/uv-cache uv run --with ruff ruff check scripts/run_rdna4_profiler_timing_smoke.py tests/sol_execbench/test_rdna4_profiler_timing_smoke.py`
  - `All checks passed!`
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_rocm_profiler.py tests/sol_execbench/test_rdna4_profiler_timing_smoke.py -q`
  - `21 passed in 4.28s`

## Usage

On the RDNA4 validation host:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run scripts/run_rdna4_profiler_timing_smoke.py \
  --output-dir out/rdna4-profiler-backed-timing-smoke \
  --timing-tool-version "rocprofv3 7.1.1" \
  --gpu-architecture gfx1200 \
  --workload-limit 1
```

## Claim Boundary

This adds a bounded profiler-backed timing smoke path only. It does not upgrade
RDNA4 to full paper validation, score authority, leaderboard readiness, broader
AMD hardware validation, CDNA3/MI300X validation, or CDNA4 validation.
