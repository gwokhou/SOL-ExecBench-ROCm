---
phase: 51-sidecar-coverage-and-score-guards
plan: 03
subsystem: testing
tags: [solar, sidecar, parser, score-guards, public-contracts]

requires:
  - phase: 51-sidecar-coverage-and-score-guards
    provides: 51-01 SOLAR coverage summary and aggregate status sidecars
  - phase: 51-sidecar-coverage-and-score-guards
    provides: 51-02 AMD-native SOLAR aggregate score guards
provides:
  - Full TEST-03 parser and serializer regression coverage for Phase 51 sidecar fields
  - Public boundary guardrails for Phase 51 internal fields across canonical schemas, CLI help, trace JSONL, and score evidence refs
  - Full Phase 49-51 regression gate verification without GPU or candidate execution
affects: [phase-51, phase-52, solar-derivation, amd-native-score, public-contracts]

tech-stack:
  added: []
  patterns:
    - Exact-key parser tests for nested internal sidecar payloads
    - Recursive public artifact key checks to avoid substring false positives

key-files:
  created:
    - .planning/phases/51-sidecar-coverage-and-score-guards/51-03-SUMMARY.md
  modified:
    - tests/sol_execbench/test_solar_derivation_evidence.py
    - tests/sol_execbench/test_public_contract_guardrails.py
    - tests/sol_execbench/test_amd_sol_v2.py

key-decisions:
  - "Keep Phase 51 coverage and aggregate fields sidecar-only; do not add public schemas, CLI flags, report fields, dependencies, GPU checks, or candidate execution."
  - "Treat AMD SOL v2 coverage_summary as an existing artifact field while guarding exact SOLAR sidecar keys from leaking into v2 payloads."

patterns-established:
  - "Coverage parser tests now mutate every Phase 51 nested field class for malformed, missing, unknown, and legacy-normalization behavior."
  - "Public guardrails check exact JSON object keys for derived artifacts where substring matching would collide with established v2 names."

requirements-completed: [REPORT-01, REPORT-02, REPORT-03, TEST-03]

duration: 5min
completed: 2026-05-23
---

# Phase 51 Plan 03: Sidecar Coverage And Score Guards Summary

**Phase 51 parser, score, and public-boundary regression gates now cover every machine-verifiable SOLAR sidecar field without changing public contracts**

## Performance

- **Duration:** 5 min
- **Started:** 2026-05-23T09:13:50Z
- **Completed:** 2026-05-23T09:19:10Z
- **Tasks:** 3
- **Files modified:** 4

## Accomplishments

- Expanded SOLAR derivation round-trip assertions to cover `coverage_summary` family/status counts, family records, missing/unsupported patterns, degraded/unsupported/estimated node IDs, provenance refs, and `aggregate_status`.
- Added malformed and strict parser tests for invalid aggregate status, non-boolean `score_eligible`, malformed count maps, non-list node IDs, malformed provenance refs, missing nested fields, unknown nested fields, exact legacy payload normalization, and unknown legacy key rejection.
- Extended public guardrails so Phase 51 internals stay absent from canonical Definition/Workload/Trace payloads, primary CLI help, canonical trace JSONL, public score `evidence_refs`, and AMD SOL v2 artifacts where they would be SOLAR-only keys.
- Verified Phase 49/50 family modeling, AMD bound graph/estimate, sidecar evidence, AMD SOL v2, public boundary, and AMD-native score tests together.

## Task Commits

1. **Task 51-03-01: Complete TEST-03 parser and round-trip matrix** - `ea66227` (`#51 - Close SOLAR coverage parser matrix`)
2. **Task 51-03-02: Lock public boundary and score regression guardrails** - `1376fee` (`#51 - Preserve public score guard boundaries`)
3. **Task 51-03-03: Run full Phase 51 regression gate** - no code commit; verification-only task completed after task commits.

## Files Created/Modified

- `tests/sol_execbench/test_solar_derivation_evidence.py` - Full TEST-03 sidecar parser, round-trip, malformed payload, exact legacy payload, unknown field, and deterministic coverage closure.
- `tests/sol_execbench/test_public_contract_guardrails.py` - Phase 51 public schema, CLI, trace JSONL, score evidence ref, and AMD SOL artifact leakage guardrails.
- `tests/sol_execbench/test_amd_sol_v2.py` - Regression assertion that AMD SOL v2 artifact semantics remain free of SOLAR derivation-only fields.
- `.planning/phases/51-sidecar-coverage-and-score-guards/51-03-SUMMARY.md` - Execution summary and verification record.

## Decisions Made

- Preserved Phase 51 as test and guardrail closure only; no public schema, CLI, canonical trace JSONL, dependency, docs, runner presentation, hardware, or candidate-execution changes were made.
- Used exact recursive JSON key checks for derived artifacts so established AMD SOL v2 keys such as `op_family_counts` are not mistaken for Phase 51 `family_counts` leakage.

## Deviations from Plan

None - plan executed within the requested 51-03 scope.

## Known Stubs

None.

## Threat Flags

None. The only touched trust-boundary surfaces were planned test guardrails for strict sidecar parsing and public artifact containment.

## Issues Encountered

- Initial public artifact guardrail used substring matching and falsely matched AMD SOL v2 `op_family_counts`; fixed the test to inspect exact JSON object keys.
- Initial provenance assertion expected a different deterministic order; updated it to match the existing stable `(group_id, node_id, tensor_id, kind, detail)` ordering.

## Verification

- `uv run pytest tests/sol_execbench/test_solar_derivation_evidence.py -k "coverage or aggregate or parser or malformed or unknown or round_trip or deterministic" -n 0 -x` - 59 passed, 16 deselected.
- `uv run pytest tests/sol_execbench/test_public_contract_guardrails.py tests/sol_execbench/test_amd_native_score.py tests/sol_execbench/test_amd_sol_v2.py -k "solar or coverage or aggregate or score or cli or trace or schema" -n 0 -x` - 32 passed, 13 deselected.
- `uv run pytest tests/sol_execbench/test_solar_derivation_family_modeling.py tests/sol_execbench/test_solar_derivation_evidence.py tests/sol_execbench/test_amd_bound_graph.py tests/sol_execbench/test_amd_bound_estimates.py tests/sol_execbench/test_public_contract_guardrails.py tests/sol_execbench/test_amd_sol_v2.py tests/sol_execbench/test_amd_native_score.py -n 0` - 189 passed.
- `uv run --with ruff ruff check src/sol_execbench/core/scoring/solar_derivation.py src/sol_execbench/core/scoring/amd_score.py tests/sol_execbench/test_solar_derivation_evidence.py tests/sol_execbench/test_amd_native_score.py tests/sol_execbench/test_public_contract_guardrails.py tests/sol_execbench/test_amd_sol_v2.py` - passed.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

Phase 51 is closed for parser, score guard, and public-boundary regression coverage. Phase 52 can address runner-facing presentation and documentation without changing the internal sidecar parser or public contract guardrails established here.

## Self-Check: PASSED

- Found `.planning/phases/51-sidecar-coverage-and-score-guards/51-03-SUMMARY.md`.
- Found task commit `ea66227`.
- Found task commit `1376fee`.

---
*Phase: 51-sidecar-coverage-and-score-guards*
*Completed: 2026-05-23*
