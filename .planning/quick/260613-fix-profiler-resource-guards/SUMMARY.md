---
status: complete
---

# Summary

Fixed RDNA4 profiler batch resource guardrails:

- Corrected ROCm GPU counting to count unique `GPU[n]` ids from `rocm-smi --showid`.
- Made RDNA4 profiler batch omit HIP runtime API tracing by default.
- Added `--keep-staging` for explicit staging-dir retention; default now cleans staging dirs.
- Added `--keep-rocprofv3-csv` for explicit raw CSV retention; default now removes rocprofv3 run dirs after compact sidecars are written.
- Made RDNA4 batch timing sidecars compact parsed rows by default.

# Verification

- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/core/bench/test_timing_isolation.py tests/sol_execbench/test_rdna4_profiler_timing_batch.py -q`
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_rocm_profiler.py tests/sol_execbench/test_rdna4_profiler_timing_coverage.py tests/sol_execbench/test_profiler_timing_coverage.py -q`
- `UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/run_rdna4_profiler_timing_batch.py --help`
- Real host isolation check now reports `gpu_count: 1` and no multi-GPU warning.
