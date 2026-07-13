---
quick_id: 260608-jan
slug: clarify-rdna4-timing-fallback-semantics-
status: complete
completed: 2026-06-08
---

# Summary

Clarified timing fallback semantics after Phase 147:

- Added HIP native source-collection coverage proving native ROCm sources route
  to profiler-backed `rocprofv3` kernel activity timing when available.
- Updated `docs/user/rocm_timing.md` to distinguish source-policy fallback from
  profiler-unavailable fallback.
- Updated `.planning/STATE.md` with the quick task record.

## Verification

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest \
  tests/sol_execbench/test_rocm_profiler.py \
  tests/sol_execbench/test_timing_policy.py -q

UV_CACHE_DIR=/tmp/uv-cache uv run --with ruff ruff check \
  src/sol_execbench/core/bench/rocm_profiler.py \
  tests/sol_execbench/test_rocm_profiler.py \
  tests/sol_execbench/test_timing_policy.py

git diff --check
```
