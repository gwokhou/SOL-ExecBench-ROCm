# Research: Stack for v1.38 Confirmed Benchmark Evidence

**Date:** 2026-06-21

## Existing Context

SOL-ExecBench-ROCm already has optional `rocprofv3` profile collection,
diagnostic `profile_summary.sidecar.v1`, derived AMD-native score reports, and
release-scoped scoring baseline artifacts. HIP now needs confirmed benchmark
evidence that is stronger than diagnostic sidecars: profile artifacts must be
discoverable, official score evidence must be distinguishable from trace
speedup, and measured baselines must carry enough provenance for coverage
validation.

## AMD Sources

- ROCprofiler-SDK `rocprofv3` supports kernel trace, HIP runtime trace, CSV,
  JSON, rocpd, Perfetto, and other output formats. Its JSON output is designed
  for programmatic analysis, which fits structured artifact discovery and
  citations.
- ROCm Compute Profiler is open source under the ROCm Systems super-repo. Its
  performance model exposes System Speed-of-Light metrics such as IPC, L2
  bandwidth/hit rate, MFMA/VALU utilization, LDS bank conflicts, and bandwidth
  percent-of-peak fields that can seed bounded bottleneck hints.
- The ROCm Systems super-repo lists `rocprofiler`, `rocprofiler-compute`,
  `rocprofiler-register`, and `rocprofiler-sdk` as public completed migrations,
  making AMD the primary source for profiler behavior.

## NVIDIA/SOL Sources

- NVIDIA's SOL-ExecBench README states leaderboard ranking is based on
  SOL-Score, which grades custom kernel performance against a theoretical B200
  roofline derived analytically by SOLAR.
- The SOL-ExecBench paper defines SOL Score as closing the gap between a
  release-defined scoring baseline and a hardware SOL bound, not simply
  diagnostic trace speedup.

## Stack Additions

- Extend `core.bench.rocm_profiler` artifact discovery and tests so successful
  `rocprofv3` profile runs cannot be classified as
  `rocprof_no_registered_artifacts` when files exist under version-specific
  output layouts.
- Extend `core.bench.profile_summary` with structured kernel/workload metrics,
  source artifact citations, and bounded bottleneck hints derived from
  `rocprofv3`/ROCm Compute Profiler compatible fields.
- Add an official SOL score evidence schema/report that records score source,
  aggregation policy, numeric score coverage, and authoritative input refs.
- Add a measured baseline evidence schema/report with trace pointer, hardware,
  ROCm/SOL version, target identity, timing policy, and workload coverage.
- Extend evaluator contract capabilities for confirmed benchmark evidence while
  leaving agent feedback and profile summary authority flags diagnostic-only.

## Source Links

- https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html
- https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/conceptual/system-speed-of-light.html
- https://github.com/ROCm/rocm-systems
- https://github.com/NVIDIA/SOL-ExecBench
- https://arxiv.org/abs/2603.19173
