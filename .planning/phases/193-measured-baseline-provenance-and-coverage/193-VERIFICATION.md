---
phase: 193
status: passed
verified_at: "2026-07-10T09:45:00Z"
verification_type: "goal-backward"
requirements: ["BASE-01", "BASE-02", "BASE-03"]
---

# Phase 193 Verification: Measured Baseline Provenance and Coverage

## Verdict

PASSED. Phase 193 adds measured baseline generation timestamp (BASE-01), a
five-state coverage validator with stable reason codes (BASE-02), and wires
measured baseline provenance into the official score gate as a coverage-gated
authoritative baseline source while keeping schemas separate and the gate
STAGING (BASE-03). All work is CPU-safe unit-tested without a real GPU.

## Goal

Add measured baseline evidence and coverage validation for confirmed benchmark
claims.

## Success Criteria

1. **Measured baseline evidence records trace pointer, hardware, ROCm version,
   SOL version, target identity, timing policy, workload coverage, and
   timestamp:** VERIFIED
   - `export_hip_baseline_registry()` (pre-existing) already records `trace_ref`,
     `provenance.{hardware,rocm_version,sol_version,target_id,timing_policy}`,
     and `expected_workload_keys` (workload coverage).
   - Plan 193-01 added the top-level `generated_at` ISO-8601 UTC timestamp,
     reusing the shared `utc_timestamp()` freshness helper, without a schema
     bump (additive `baseline_registry.v1`).
   - `test_export_hip_baseline_registry_from_passed_trace` and
     `test_export_hip_baseline_registry_generated_at_override` pin the field and
     its injectability.

2. **Coverage validation reports confirmed, missing, stale, mismatched, and
   placeholder baseline states with stable reason codes:** VERIFIED
   - New `src/sol_execbench/core/evidence/baseline_coverage.py` classifies each
     expected workload into exactly one of the five states.
   - Reason codes: `missing_baseline` / `placeholder_baseline` (reused official
     score literals, D-08) for missing/placeholder; new `baseline_stale` +
     `baseline_stale_trace`, `baseline_mismatched` +
     `baseline_hardware_mismatch` / `baseline_timing_policy_mismatch` (D-06/D-07)
     for stale/mismatched; `confirmed` is a positive status, not a blocker
     (D-09).
   - The validator takes an explicit `CurrentRunEnvironment` (D-03) and an
     optional `trace_root` for relative trace-ref resolution.

3. **Baseline evidence integrates with official score generation without treating
   `reference_latency_ms` as release-defined baseline evidence:** VERIFIED
   - `DEFAULT_OFFICIAL_BASELINE_SOURCES` now accepts both `scoring_baseline` and
     `measured_baseline_registry` (D-10/D-13 — schemas kept separate).
   - The gate accepts an optional `coverage_report` precondition; a non-confirmed
     report adds the `baseline_coverage_failed` umbrella blocker plus the
     report's specific reason codes and forces the official score to `None`
     (D-11). `coverage_report=None` preserves prior behavior (backward
     compatible).
   - `reference_latency_ms` and placeholder/reference baseline fallback remain
     blocked for official claims (Phase 192 D-07/D-08 unchanged).
   - End-to-end sanity: confirmed coverage → score 0.75, no blockers; mismatched
     coverage → score None, blockers `('baseline_coverage_failed',
     'baseline_mismatched', 'baseline_hardware_mismatch')`.

4. **Tests cover complete coverage, partial coverage, hardware mismatch, timing
   policy mismatch, stale trace pointer, and placeholder baseline rejection:**
   VERIFIED
   - `test_baseline_coverage.py` has all six required cases plus `to_dict` shape,
     confirmed-is-not-a-blocker, and report-type tests (8 tests).
   - `test_official_score_evidence.py` adds measured-baseline-registry
     acceptance, coverage-failure propagation, `coverage_report=None` backward
     compatibility, both-sources-accepted, blocker-literal consistency (D-08),
     and suite-level coverage blocking (6 new tests).

## Requirements Coverage

- **BASE-01**: Plan 193-01 — `generated_at` timestamp on the registry.
- **BASE-02**: Plan 193-02 — `baseline_coverage.py` five-state validator.
- **BASE-03**: Plan 193-03 — official score gate wiring with schemas separate.

## Test Results

- `tests/sol_execbench/core/evidence/` + `tests/sol_execbench/core/scoring/`:
  381 passed, 0 failed.
- Focused: `test_baseline_export.py` (7), `test_baseline_coverage.py` (8),
  `test_official_score_evidence.py` (14), `test_public_import_facades.py` +
  `test_baseline.py` (6) — all pass.
- Ruff check + format clean on all changed files.

## Authority Boundaries Preserved

- `official_score.py` remains STAGING — NOT wired into any CLI/runner/sidecar
  (verified: no `official_score` references in `src/sol_execbench/cli/` or
  `src/sol_execbench/driver/`). Emission belongs to Phase 194 (GATE-01).
- Canonical Trace JSONL authority unchanged; coverage module is a classifier
  only and does not import from `sol_execbench.core.scoring`.
- `ScoringBaselineArtifact` (release-scoped) and the measured baseline registry
  remain separate schemas.

## Residual Risk / Hand-off

- Full `uv run pytest tests/` is not run (host env blockers noted in STATE.md:
  missing `triton`, legacy report scripts, Docker shell compatibility). Focused
  evidence/scoring suites pass cleanly.
- HIP-facing confirmed/missing/placeholder/profiler-partial/diagnostic-only
  fixtures and consumer docs are Phase 194 (GATE-02).
- Official score emission into a run path is Phase 194 (GATE-01); the
  aggregation-policy precondition remains an explicit parameter.
