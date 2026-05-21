# 06-02 Summary: Schema, Trace, And Analysis Docs

## Completed

- Rewrote `docs/solution.md` around ROCm schema values, HIP compile options, and replacement guidance.
- Rewrote `docs/trace.md` with AMD/ROCm environment examples.
- Updated `docs/definition.md` device wording from CUDA-specific to GPU/ROCm-compatible.
- Added `docs/analysis.md` for trace review, timing, clock locking, and `rocprofv3` profiling.
- Updated the trace model docstring to use an AMD hardware example.

## Evidence

- `uv run --no-sync pytest tests/sol_execbench/core/data/test_solution.py tests/sol_execbench/test_rocm_schema_build_audit.py` -> 59 passed.
- `uv run --no-sync ruff check src tests` -> passed.
