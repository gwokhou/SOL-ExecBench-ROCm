# Phase 32 Context: Source-Specific Profiler Timing Workflow

**Date:** 2026-05-22
**Status:** Complete

## Problem

The ROCm port had pure timing policy helpers and low-level `rocprofv3` parsing,
but dataset execution did not expose an end-to-end source-specific timing
workflow. Timing evidence also lacked benchmark run metadata needed to audit
profiler numbers against event timing traces.

## Relevant Code

- `src/sol_execbench/core/bench/timing_policy.py` classifies source languages
  and selects explicit timing backends.
- `src/sol_execbench/core/bench/rocm_profiler.py` builds `rocprofv3` commands,
  parses CSV output, and returns fallback metadata.
- `scripts/run_dataset.py` owns dataset batch execution and optional derived
  reporting.
- `docs/internal/analysis.md` documents benchmark trace and profiling workflows.

## Constraints

- Canonical trace JSONL remains unchanged.
- Profiler-backed timing must be opt-in and must label fallback explicitly.
- Tests must not require installed ROCm profiler tools or GPU hardware.
