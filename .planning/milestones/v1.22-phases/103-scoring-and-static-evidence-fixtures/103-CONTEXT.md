# Phase 103: Scoring And Static Evidence Fixtures - Context

**Gathered:** 2026-06-01
**Status:** Ready for planning

## Phase Boundary

Maintainers can validate scoring derivation and static-evidence changes against
focused fixtures without changing public contracts or diagnostic authority.

This phase covers requirements SCORING-01 through SCORING-04:

- Add focused family-specific golden checks for SOLAR and AMD bound derivation.
- Cover confidence/status transitions independently from broad report-shape
  tests.
- Let static kernel evidence consume or produce an explicit artifact manifest
  when build outputs are known.
- Preserve diagnostic-only authority and existing public sidecar contracts.

## Current Code Context

- `src/sol_execbench/core/scoring/solar_derivation.py` already exposes
  `classify_solar_confidence()` as a pure status decision helper and
  `build_solar_derivation_evidence()` / `derive_solar_derivation_evidence()`
  for sidecar construction.
- `src/sol_execbench/core/scoring/amd_bound_estimates.py` dispatches
  operator-family estimates through `estimate_dispatch_family()` and
  `estimate_bound_work()`.
- `src/sol_execbench/core/scoring/amd_sol_v2.py` exposes the derived AMD SOL v2
  sidecar with per-family confidence counts and aggregate status transitions.
- Existing tests already cover representative matmul, inexact, unsupported,
  MoE, attention, SSM, and sidecar parser behavior, but some of that coverage
  is embedded in broad fixture/sidecar tests rather than small golden fixtures.
- `src/sol_execbench/core/bench/static_kernel_evidence.py` discovers artifacts
  by scanning a build root for the primary shared object plus supported static
  artifact suffixes. It does not yet consume an explicit build artifact
  manifest, which is the preferred future boundary for known build outputs.

## Implementation Direction

Add focused CPU-safe tests that lock representative family formulas and
confidence/status transitions without changing public scoring schemas. Add an
optional manifest path to static evidence collection; when provided, collection
should use root-contained manifest entries instead of recursive discovery and
record manifest provenance in sidecar source references.

Keep all authority fields and sidecar schemas unchanged. Manifest support should
be additive and optional.

## Verification Targets

- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_amd_bound_estimates.py tests/sol_execbench/test_amd_sol_v2.py tests/sol_execbench/test_static_kernel_evidence.py -q`
- `UV_CACHE_DIR=/tmp/uv-cache uv run ruff check src/sol_execbench/core/bench/static_kernel_evidence.py tests/sol_execbench/test_static_kernel_evidence.py tests/sol_execbench/test_amd_bound_estimates.py tests/sol_execbench/test_amd_sol_v2.py`

