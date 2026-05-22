# Phase 28 Research: Live rocprofv3 Timing Integration

**Phase:** 28 - Live rocprofv3 Timing Integration
**Researched:** 2026-05-22
**Status:** Ready for planning

## Current Implementation

`rocm_profiler.py` currently contains policy-aware helpers but does not execute
commands:

- `build_rocprofv3_command()`
- `parse_rocprofv3_csv()`
- `build_timing_evidence()`
- `select_default_timing()`

`timing_policy.py` already separates HIP native, Triton, PyTorch, mixed, and
unknown source semantics.

## Recommended Implementation

Add a live collection adapter that:

- accepts an application command, timing policy, output directory, output file,
  tool version, and GPU architecture;
- uses `build_rocprofv3_command()` for HIP native and Triton policies when
  profiler-backed timing is selected;
- runs the command through an injectable runner for unit-test mocking;
- reads the generated CSV evidence and returns `Rocprofv3TimingEvidence`;
- returns explicit fallback evidence when profiler-backed collection is not
  selected or not available.

## Testing Strategy

Unit tests should not require real hardware or `rocprofv3`. Use an injected
runner that writes fixture CSV output and returns a completed process-like
object. Test fallback paths without invoking subprocesses.

## Guardrails

- Do not change `time_runnable()` default behavior.
- Do not add fields to canonical trace JSONL.
- Keep PyTorch attribution separate from raw kernel activity.
- Keep compile/autotune/warmup/unrelated rows excluded or labeled by evidence
  policy rather than silently included.
