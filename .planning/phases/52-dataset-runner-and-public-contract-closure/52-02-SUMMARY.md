---
phase: 52-dataset-runner-and-public-contract-closure
plan: 02
subsystem: docs
tags: [docs, solar-derivation, claim-boundaries, dataset-runner]

requires:
  - phase: 52-dataset-runner-and-public-contract-closure
    provides: "52-01 dataset-runner SOLAR sidecar/report integration"
provides:
  - "Documented v1.10 dataset-runner SOLAR sidecar workflow"
  - "Documented AMD-native-derived claim boundaries for v1.10 derived reports"
  - "Documented derivation source boundary excluding candidate execution"
affects: [phase-52, REPORT-04, TEST-05, docs]

tech-stack:
  added: []
  patterns:
    - "Update existing analysis/internal docs instead of adding a broad new tutorial"
    - "Use explicit no-claim language rather than broad forbidden-token wording"

key-files:
  created:
    - ".planning/phases/52-dataset-runner-and-public-contract-closure/52-02-SUMMARY.md"
  modified:
    - "docs/analysis.md"
    - "docs/internal/solar_derivation_contract.md"

key-decisions:
  - "Document the exact `--solar-derivation` option from Plan 52-01 alongside existing AMD score and bound sidecar options."
  - "Describe `derived_evidence_refs` as derived-report-only metadata, not canonical trace or public evidence-ref drift."
  - "State that v1.10 is AMD ROCm derived evidence and not paper-scale extraction, upstream SOLAR parity, NVIDIA B200/Blackwell equivalence, hosted leaderboard readiness, or new hardware validation."

requirements-completed: [REPORT-04, TEST-05]

duration: manual takeover after stalled executor
completed: 2026-05-23
---

# Phase 52 Plan 02: Documentation And Claim Boundary Summary

**v1.10 SOLAR sidecar consumption and claim boundaries are documented for the dataset-runner derived report path**

## Accomplishments

- Added a local v1.10 workflow in `docs/analysis.md` using `--amd-score-report`, `--amd-sol-bound-dir`, and `--solar-derivation`.
- Explained canonical `traces.json`, AMD SOL v2 sidecars, SOLAR derivation sidecars, AMD-native score reports, `coverage_summary`, `aggregate_status`, warnings, score eligibility, and `derived_evidence_refs`.
- Clarified that `derived_evidence_refs` are derived-report-only metadata and do not change canonical trace JSONL or public score `evidence_refs`.
- Tightened `docs/internal/solar_derivation_contract.md` with explicit v1.10 claim boundaries and a derivation source boundary.
- Stated that SOLAR formula evidence derives from `Definition`, `Workload`, reference-visible, and static evidence, not candidate solution execution or benchmark timing.

## Deviations

The original executor stalled after writing the doc changes. I closed that agent, verified the edits in the main thread, added this summary, and will commit the completed plan directly.

## Verification

- `rg -n -- '--amd-score-report|coverage_summary|aggregate_status|score eligibility|amd-native-derived|canonical trace JSONL' docs/analysis.md` - passed.
- `rg -n -- 'not .*paper|not .*124-model|not .*235-problem|not .*B200|not .*Blackwell|not .*leaderboard|not .*real-hardware|NVFP4|MXFP4|paper-aligned automatic SOLAR derivation evidence' docs/analysis.md docs/internal/solar_derivation_contract.md` - passed.
- `rg -n -- 'Definition|Workload|candidate_solution_execution|candidate solution execution|static evidence|reference-visible' docs/analysis.md docs/internal/solar_derivation_contract.md` - passed.
- `! rg -n -- 'derive.*candidate|candidate.*derive|submitted.*derive|benchmark timing.*formula' docs/analysis.md docs/internal/solar_derivation_contract.md` - passed.
- `uv run pytest tests/sol_execbench/test_solar_derivation_contract.py tests/sol_execbench/test_v1_9_validation_closure.py -k "claim or boundary or paper or leaderboard or validation or B200 or Blackwell" -n 0 -x` - 6 passed, 15 deselected.

## Risks

- Final automated public-contract and claim guardrail closure remains Plan 52-03.
- This plan did not run hardware validation or dataset-scale extraction; those remain explicitly out of scope.

---
*Phase: 52-dataset-runner-and-public-contract-closure*
*Completed: 2026-05-23*
