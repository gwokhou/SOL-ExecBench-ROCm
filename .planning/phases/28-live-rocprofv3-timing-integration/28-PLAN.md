# Phase 28 Plan: Live rocprofv3 Timing Integration

**Created:** 2026-05-22
**Status:** Ready for execution
**Requirements:** PROF-01, PROF-02, PROF-03, PROF-04

## Goal

Collect live source-specific ROCm profiler timing evidence during benchmark or
dataset execution without changing canonical trace JSONL or primary CLI
defaults.

## Tasks

### 28-01: Add live collection adapter

**Files:**
- `src/sol_execbench/core/bench/rocm_profiler.py`
- `tests/sol_execbench/test_rocm_profiler.py`

**Work:**
- Add a live collection request/result model.
- Add a function that invokes `rocprofv3` through an injectable subprocess
  runner and reads generated CSV evidence.
- Preserve deterministic output directory/file handling.

### 28-02: Add fallback evidence path

**Files:**
- `src/sol_execbench/core/bench/rocm_profiler.py`
- `tests/sol_execbench/test_rocm_profiler.py`

**Work:**
- Return explicit fallback metadata when `rocprofv3` is unavailable or when
  policy selects PyTorch, mixed, unknown, or event fallback semantics.
- Preserve source-specific policy interpretation in every payload.

### 28-03: Add live collector tests

**Files:**
- `tests/sol_execbench/test_rocm_profiler.py`

**Work:**
- Test command construction and mocked subprocess invocation.
- Test CSV file discovery/readback and evidence payload contents.
- Test non-profiler PyTorch/mixed fallback does not masquerade as kernel
  activity.

### 28-04: Document live timing boundary

**Files:**
- `docs/rocm_timing.md`

**Work:**
- Describe live adapter behavior, output boundaries, and source-specific
  chimney semantics.
- State that canonical trace JSONL remains unchanged.

## Verification

- `uv run pytest tests/sol_execbench/test_rocm_profiler.py tests/sol_execbench/test_timing_policy.py`
- `uv run ruff check src/sol_execbench/core/bench/rocm_profiler.py tests/sol_execbench/test_rocm_profiler.py`

## Requirement Mapping

| Requirement | Task |
|-------------|------|
| PROF-01 | 28-01 |
| PROF-02 | 28-01, 28-02 |
| PROF-03 | 28-02, 28-03 |
| PROF-04 | 28-01, 28-04 |
