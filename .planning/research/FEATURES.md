# Research: Features for v1.38 Confirmed Benchmark Evidence

**Date:** 2026-06-21

## Table Stakes

- Successful requested `rocprofv3` profile runs register at least one concrete
  profiler artifact with kind, path, size, checksum where practical, and compact
  citation in profile summary.
- Profile summary contains structured profiling evidence, not only status:
  workload identity, kernel rows, duration/dispatch metrics, artifact coverage,
  and bottleneck hints mapped to stable categories.
- Official score evidence is non-null for valid runs and exposes score source,
  aggregation policy, scored/unscored counts, and authoritative input refs.
- Measured baseline evidence exposes target identity, trace pointer, hardware,
  ROCm version, SOL version, timing policy, workload coverage, and blockers.
- HIP can validate confirmed coverage without scraping diagnostic sidecar prose.

## Differentiators

- Keep profile summary and agent feedback diagnostic-only, while adding a
  separate confirmed benchmark evidence surface for score/baseline/cutover
  decisions.
- Prefer AMD profiler semantics and metric names, but translate into a compact
  SOL-owned taxonomy that HIP can consume without depending on raw profiler
  dumps.
- Make invalid runs explicit through blockers such as missing profiler
  artifacts, missing score evidence, missing baseline coverage, unsupported SOL
  bounds, or placeholder/reference-only baseline use.

## Deferred

- Claiming paper parity, leaderboard readiness, NVIDIA B200 equivalence, or
  full CDNA3/CDNA4 validation.
- Replacing ROCm Compute Profiler; this milestone should parse/normalize
  evidence compatible with available `rocprofv3` artifacts and bounded metrics.
- Letting diagnostic sidecars become score, release-gate, cutover, or
  confirmed-improvement authority.
