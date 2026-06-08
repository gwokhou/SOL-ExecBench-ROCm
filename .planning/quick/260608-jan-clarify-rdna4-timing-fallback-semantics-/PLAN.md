---
quick_id: 260608-jan
slug: clarify-rdna4-timing-fallback-semantics-
status: in_progress
created: 2026-06-08
---

# Clarify RDNA4 Timing Fallback Semantics

## Goal

Make timing fallback semantics unambiguous after Phase 147: PyTorch
device-event fallback is a source-policy boundary, not evidence that
`rocprofv3` is missing or broken.

## Tasks

1. Add regression coverage proving HIP native source timing still routes to
   profiler-backed `rocprofv3` collection when available.
2. Tighten timing documentation to distinguish source-policy fallback from
   tool-unavailable or profiler-failure fallback.
3. Record quick-task verification and update project state.

## Verification

- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_rocm_profiler.py tests/sol_execbench/test_timing_policy.py -q`
- `UV_CACHE_DIR=/tmp/uv-cache uv run --with ruff ruff check src/sol_execbench/core/bench/rocm_profiler.py tests/sol_execbench/test_rocm_profiler.py tests/sol_execbench/test_timing_policy.py`
- `git diff --check`
