---
phase: 87-amd-sol-solar-bound-sanity-evidence
verified: 2026-05-31T11:32:05Z
status: passed
score: 9/9 must-haves verified
overrides_applied: 0
re_verification:
  previous_status: passed_with_review_blocker
  previous_score: 8/9
  blocker_commit: "9857647"
  gaps_closed:
    - "Source refs with nested checksum objects are normalized to checksum.value instead of Python dict strings."
  gaps_remaining: []
  regressions: []
---

# Phase 87: AMD SOL/SOLAR Bound Sanity Evidence Verification Report

**Phase Goal:** Researchers can generate diagnostic AMD SOL/SOLAR bound sanity reports over existing RDNA 4 and Docker evidence while keeping score eligibility and hardware-validation claims unchanged.
**Verified:** 2026-05-31T11:32:05Z
**Status:** passed
**Re-verification:** Yes - after checksum blocker fix commit `9857647`

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Researcher can generate `amd_bound_sanity.v1` over existing trace, closure, AMD SOL, SOLAR derivation, AMD score, and compatibility evidence refs/checksums. | VERIFIED | `AMD_BOUND_SANITY_SCHEMA_VERSION` remains `sol_execbench.amd_bound_sanity.v1`; `build_amd_bound_sanity_report()` emits strict checked reports and source refs. The prior blocker is closed: `_source_from_ref()` now uses `_checksum(value)`, and `test_sanity_source_refs_normalize_nested_checksum_values` proves `{"checksum": {"value": "trace-sha"}}` becomes `"trace-sha"`. |
| 2 | Researcher can inspect artifact availability, aggregate statuses, coverage summaries, warnings, and evidence gaps for AMD SOL/SOLAR evidence. | VERIFIED | Report fields include `artifact_availability`, `amd_sol_aggregate_statuses`, `solar_aggregate_statuses`, `coverage_summary`, `warnings`, and `evidence_gaps`; core tests assert these fields. |
| 3 | Researcher can distinguish scored, degraded, unscored, unsupported, provisional, and missing-evidence states without changing AMD-native score semantics or score eligibility rules. | VERIFIED | `SANITY_STATUS_KEYS` contains the full diagnostic vocabulary, rollup preserves source `amd_score_supported`, and AMD score guardrails continue to pass. |
| 4 | Bound sanity output surfaces provisional RDNA 4 model risk while explicitly avoiding upstream SOLAR equivalence, model-validation, paper-parity, leaderboard, CDNA 3, MI300X, CDNA 4, and native-host validation claims. | VERIFIED | `AmdBoundSanityClaimBoundary` keeps all authority/validation fields false except diagnostic provisional risk; Markdown guardrails assert the negative wording remains visible. |
| 5 | Bound sanity checks run without new hardware probes, Docker privilege changes, or dependency relocking. | VERIFIED | Verification ran CPU-safe pytest and script help only. No GPU/ROCm/Docker probes were run. `git diff` over `pyproject.toml`, `uv.lock`, canonical data models, AMD score, and primary CLI was empty. |
| 6 | Researcher can generate JSON and Markdown through a thin script wrapper over the core report builder. | VERIFIED | `scripts/report_amd_bound_sanity.py` loads only explicit JSON paths, delegates to `build_amd_bound_sanity_report()` and `write_amd_bound_sanity_reports()`, and script tests passed. |
| 7 | The tool remains sidecar/reporting infrastructure and is not exposed as a primary `sol-execbench` benchmark CLI option. | VERIFIED | No primary CLI or entry-point diff; `test_primary_cli_does_not_expose_v1_19_amd_bound_sanity_options` passed. |
| 8 | Canonical Trace, Definition, Workload, Solution, and AMD-native score eligibility contracts are unchanged. | VERIFIED | Public contract guardrails passed; no tracked diff exists in canonical data model paths, `amd_score.py`, or `src/sol_execbench/cli/main.py`. |
| 9 | SANITY-01 through SANITY-04 are covered by CPU-safe tests and implementation evidence. | VERIFIED | Required tests plus adjacent AMD SOL/SOLAR/score tests passed: 158 tests. |

**Score:** 9/9 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/sol_execbench/core/scoring/amd_bound_sanity.py` | Strict report models, source-ref normalization, rollups, checksums, Markdown, write helpers | VERIFIED | Exists, substantive, wired to checksum helper; checksum blocker fixed in `9857647`. |
| `scripts/report_amd_bound_sanity.py` | Thin argparse wrapper from existing artifact paths | VERIFIED | Exists and script help works; no benchmark/GPU/Docker execution path. |
| `tests/sol_execbench/test_amd_bound_sanity.py` | CPU-safe SANITY-01..04 core and checksum regression tests | VERIFIED | Includes nested checksum regression and no-probe guard. |
| `tests/sol_execbench/test_amd_bound_sanity_script.py` | CPU-safe script tests | VERIFIED | Verifies deterministic JSON/Markdown from explicit sidecar paths and missing optional evidence gaps. |
| `tests/sol_execbench/test_public_contract_guardrails.py` | Public contract guardrails | VERIFIED | Verifies sidecar-only fields, AMD score exclusion, primary CLI non-exposure, and negative boundary wording. |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `amd_bound_sanity.py` | AMD SOL/SOLAR/AMD score evidence | Source status, coverage, warnings, support flags | VERIFIED | `gsd-sdk query verify.key-links` passed for Plan 87-01. |
| `amd_bound_sanity.py` | `stable_json_checksum` | Report checksum generation | VERIFIED | `AmdBoundSanityReport.with_checksum()` uses `stable_json_checksum`. |
| `report_amd_bound_sanity.py` | `amd_bound_sanity.py` | `load_json`, builder, writer | VERIFIED | `gsd-sdk query verify.key-links` passed for Plan 87-02. |
| Public guardrails | Canonical data, AMD score, primary CLI | Serialization and Click help assertions | VERIFIED | Guardrail tests passed. |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|--------|
| `amd_bound_sanity.py` | `sources`, `artifact_availability`, `workloads`, `status_totals`, `evidence_gaps`, `claim_boundary` | Existing dictionaries/refs supplied to `build_amd_bound_sanity_report()` | Yes | VERIFIED |
| `scripts/report_amd_bound_sanity.py` | `report` | Explicit CLI JSON paths loaded by `load_json()` and passed to core builder | Yes | VERIFIED |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Focused CPU-safe suite covers checksum fix, core, script, public guardrails, AMD-native score, AMD SOL, and SOLAR derivation evidence | `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_amd_bound_sanity.py tests/sol_execbench/test_amd_bound_sanity_script.py tests/sol_execbench/test_public_contract_guardrails.py tests/sol_execbench/test_amd_native_score.py tests/sol_execbench/test_amd_sol_bounds.py tests/sol_execbench/test_solar_derivation_evidence.py -q` | `158 passed in 4.82s` | PASS |
| Script is runnable without benchmark/GPU/Docker paths | `UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/report_amd_bound_sanity.py --help` | Help displayed explicit sidecar-path options | PASS |
| No dependency/canonical/primary CLI tracked diffs | `git diff -- src/sol_execbench/core/scoring/amd_bound_sanity.py tests/sol_execbench/test_amd_bound_sanity.py scripts/report_amd_bound_sanity.py tests/sol_execbench/test_amd_bound_sanity_script.py tests/sol_execbench/test_public_contract_guardrails.py pyproject.toml uv.lock src/sol_execbench/core/data src/sol_execbench/core/scoring/amd_score.py src/sol_execbench/cli/main.py` | Empty diff after `9857647` | PASS |

### Probe Execution

| Probe | Command | Result | Status |
|-------|---------|--------|--------|
| GPU/ROCm/Docker probes | Not run | Explicitly skipped per user scope. Verification used CPU-safe tests and static checks only. | SKIPPED |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| SANITY-01 | 87-01, 87-02 | Emit `amd_bound_sanity.v1` over existing evidence with availability, statuses, coverage, warnings, gaps, refs/checksums | SATISFIED | Builder/report/script tests passed; nested checksum blocker regression passed. |
| SANITY-02 | 87-01, 87-02 | Distinguish scored/degraded/unscored/unsupported/provisional/missing-evidence without score eligibility changes | SATISFIED | Core status tests and AMD-native score tests passed. |
| SANITY-03 | 87-01, 87-02 | Surface provisional RDNA 4 risk and avoid equivalence/validation/authority/hardware claims | SATISFIED | Claim boundary and Markdown guardrails passed. |
| SANITY-04 | 87-01, 87-02 | Consume existing trace/closure/AMD SOL/SOLAR/score/compatibility refs/checksums without probes, Docker changes, or relock | SATISFIED | Checksum regression passed; no probes run; no dependency lock diff. |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| None | - | - | - | No unresolved `TBD`, `FIXME`, `XXX`, placeholder, or stub blocker found in Phase 87 touched files. Existing `subprocess.run` mention is an intentional no-probe monkeypatch in tests. |

### Human Verification Required

None.

### Gaps Summary

No blocking gaps remain. The checksum blocker from review is fixed by `9857647`, covered by a regression test, and verified in the focused CPU-safe suite. Phase 87 is final PASS with no GPU/ROCm/Docker probes, no dependency relock, no business-code edits during this re-verification, and no remaining human verification items.

---

_Verified: 2026-05-31T11:32:05Z_
_Verifier: the agent (gsd-verifier)_
