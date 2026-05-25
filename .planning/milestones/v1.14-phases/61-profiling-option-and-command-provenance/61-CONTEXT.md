# Phase 61 Context: Profiling Option and Command Provenance

## Discussion

v1.14 profiling must be opt-in and backward-compatible. The CLI can expose a
new `--profile rocprofv3` selector because profiling is explicitly disabled by
default and all existing trace JSONL, correctness, timing, and scoring schemas
remain unchanged.

The profiler command must be auditable: record the exact command, working
directory, timeout, output directory, output file prefix, and availability
state. If `rocprofv3` is unavailable, the benchmark should continue on the
normal execution path and write diagnostic metadata when an output path exists.

## Scope

- Add profile selection to the primary benchmark CLI.
- Add reusable command construction for diagnostic profiler collection.
- Record unavailable/skipped states without requiring ROCm profiler tools in CI.

## Compatibility

The new option is additive. Default execution remains equivalent to v1.13.
Profiling evidence is advertised as an optional capability and does not bump
contract version `1.0`.
