# Research Summary: v1.38 Confirmed Benchmark Evidence

**Date:** 2026-06-21

## Summary

AMD provides the right primary sources for the profiling half of this milestone:
`rocprofv3` produces programmatic profiler output and ROCm Compute Profiler
defines practical System Speed-of-Light metrics and bottleneck categories for
AMD Instinct hardware. NVIDIA/SOL remains the reference for official score
semantics: SOL Score closes the gap between a release-defined scoring baseline
and a hardware SOL bound, rather than reporting diagnostic speedup alone.

## Stack Additions

- Version-aware `rocprofv3` artifact discovery and citation coverage.
- Enriched `profile_summary.sidecar.v1` diagnostic content with structured
  workload/kernel metrics and bounded bottleneck hints.
- Separate official score evidence with source, aggregation policy, coverage,
  and input refs.
- Separate measured baseline evidence with provenance and coverage validation.
- HIP-facing contract, fixtures, and docs for confirmed pass/fail cutover
  decisions.

## Table Stakes

- Real GPU profile runs no longer report missing registered artifacts when
  profiler files exist.
- Profile summary cites concrete artifacts and carries actionable bottleneck
  evidence while keeping all authority flags diagnostic-only.
- Valid runs produce non-null official score evidence with clear source and
  aggregation policy.
- Valid runs produce measured baseline evidence with trace pointer, hardware,
  ROCm/SOL version, target identity, timing policy, and workload coverage.
- HIP can remove `missing_score`, `missing_baseline`, and
  `placeholder_baseline` blockers for valid runs.

## Watch Outs

- Do not promote profile summary or agent feedback to score/cutover authority.
- Do not treat reference latency fallback as an official measured baseline.
- Do not claim paper parity, leaderboard readiness, NVIDIA B200 equivalence,
  CDNA3 full-suite validation, or CDNA4 validation.

## Primary Sources

- ROCprofiler-SDK `rocprofv3` usage and JSON output:
  https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html
- ROCm Compute Profiler System Speed-of-Light metrics:
  https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/conceptual/system-speed-of-light.html
- ROCm Systems source-of-truth migration:
  https://github.com/ROCm/rocm-systems
- NVIDIA SOL-ExecBench README:
  https://github.com/NVIDIA/SOL-ExecBench
- SOL-ExecBench technical report:
  https://arxiv.org/abs/2603.19173
