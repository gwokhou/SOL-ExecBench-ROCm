# Phase 150 Verification

## Commands

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_profiler_timing_coverage.py tests/sol_execbench/test_rdna4_profiler_timing_batch.py -q
```

Result: `11 passed`.

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --with ruff ruff check src/sol_execbench/core/dataset/profiler_timing_coverage.py scripts/run_rdna4_profiler_timing_batch.py scripts/run_rdna4_profiler_timing_coverage.py tests/sol_execbench/test_profiler_timing_coverage.py tests/sol_execbench/test_rdna4_profiler_timing_batch.py
```

Result: `All checks passed!`

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_rocm_profiler.py tests/sol_execbench/test_profiler_timing_coverage.py tests/sol_execbench/test_rdna4_profiler_timing_batch.py -q
```

Result: `32 passed`.

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run scripts/run_rdna4_profiler_timing_coverage.py --output-dir out/rdna4-profiler-backed-timing-phase150-classification/coverage --timing-evidence-dir out/rdna4-profiler-backed-timing-batch-real-full-smoke-v2/timing --timing-evidence-dir out/rdna4-timing-evidence/timing
```

Result: exit `0`.

## Real Artifact Replay

The layered coverage replay over the existing Phase 149 real full-problem smoke
artifact produced:

- Problem denominator: `235`
- Profiler-backed problems: `0`
- Partial profiler-backed problems: `1`
- Profiler-blocked problems: `0`
- Fallback timing problems: `120`
- Readiness-blocked problems: `114`
- Full profiler-backed timing coverage: `false`

The classified partial problem is
`L1/002_vae_conv3x3_groupnorm_silu_residual_fused`, matching the Phase 149
`rocprofv3` run with 19 `PASSED` traces and 1 `INVALID_REFERENCE` trace.
