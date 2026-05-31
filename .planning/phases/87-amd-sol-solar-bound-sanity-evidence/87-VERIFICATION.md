---
phase: 87-amd-sol-solar-bound-sanity-evidence
verified: 2026-05-31T11:26:03Z
status: passed
score: 9/9 must-haves verified
overrides_applied: 0
---

# Phase 87: AMD SOL/SOLAR Bound Sanity Evidence Verification Report

**Phase Goal:** Researchers can generate diagnostic AMD SOL/SOLAR bound sanity reports over existing RDNA 4 and Docker evidence while keeping score eligibility and hardware-validation claims unchanged.
**Verified:** 2026-05-31T11:26:03Z
**Status:** passed
**Re-verification:** No - initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Researcher can generate `amd_bound_sanity.v1` over existing trace, closure, AMD SOL, SOLAR derivation, AMD score, and compatibility evidence refs/checksums. | VERIFIED | `AMD_BOUND_SANITY_SCHEMA_VERSION` is `sol_execbench.amd_bound_sanity.v1`; `build_amd_bound_sanity_report()` accepts existing dictionaries/refs, normalizes `sources`, records checksums via `_source()`/`_source_from_ref()`, and emits `AmdBoundSanityReport.with_checksum()` using `stable_json_checksum`. Script wrapper reads only explicit JSON paths and delegates to the builder. |
| 2 | Researcher can inspect artifact availability, aggregate statuses, coverage summaries, warnings, and evidence gaps for AMD SOL/SOLAR evidence. | VERIFIED | Report model includes `artifact_availability`, `amd_sol_aggregate_statuses`, `solar_aggregate_statuses`, `coverage_summary`, `warnings`, and `evidence_gaps`; Markdown renderer emits sections for each. Tests assert these fields in `test_sanity_01_report_builds_existing_evidence_summary`. |
| 3 | Researcher can distinguish scored, degraded, unscored, unsupported, provisional, and missing-evidence states without changing AMD-native score semantics or score eligibility rules. | VERIFIED | `SANITY_STATUS_KEYS` contains the required vocabulary and rollup keeps `amd_score_supported` in source evidence. `test_sanity_02_diagnostic_statuses_do_not_recompute_score_eligibility` verifies all six statuses and preservation of `supported`. Guardrails verify bound sanity fields do not enter AMD score contracts. |
| 4 | Bound sanity output surfaces provisional RDNA 4 model risk while explicitly avoiding upstream SOLAR equivalence, model-validation, paper-parity, leaderboard, CDNA 3, MI300X, CDNA 4, and native-host validation claims. | VERIFIED | `AmdBoundSanityClaimBoundary` has `provisional_rdna4_model_risk` plus false authority/validation fields. `CLAIM_BOUNDARY_TEXT` and Markdown output state the required negative boundaries, including new-hardware validation. Tests assert the boundary booleans and visible wording. |
| 5 | Bound sanity checks run without new hardware probes, Docker privilege changes, or dependency relocking. | VERIFIED | Core module imports only stdlib, Pydantic, and local checksum/manifest helpers; script uses argparse/path loading only. Tests monkeypatch `subprocess.run` to fail if called. No diff exists in `pyproject.toml` or `uv.lock`, and no Docker/ROCm/GPU probe command was run during verification. |
| 6 | Researcher can generate JSON and Markdown through a thin script wrapper over the core report builder. | VERIFIED | `scripts/report_amd_bound_sanity.py` parses explicit sidecar paths, calls `load_json()`, `build_amd_bound_sanity_report()`, and `write_amd_bound_sanity_reports()`. `test_sanity_01_04_script_writes_json_and_markdown_from_existing_paths` verifies deterministic output. |
| 7 | The tool remains sidecar/reporting infrastructure and is not exposed as a primary `sol-execbench` benchmark CLI option. | VERIFIED | No `pyproject.toml` entry point or `src/sol_execbench/cli/main.py` wiring was added. `test_primary_cli_does_not_expose_v1_19_amd_bound_sanity_options` asserts primary CLI help lacks the report options. |
| 8 | Canonical Trace, Definition, Workload, Solution, and AMD-native score eligibility contracts are unchanged. | VERIFIED | `test_v1_19_amd_bound_sanity_fields_remain_sidecar_only` asserts canonical model dumps exclude bound sanity fields; `test_v1_19_amd_bound_sanity_does_not_enter_amd_score_contracts` asserts AMD score payloads exclude the new schema and diagnostic fields. `git diff -- pyproject.toml uv.lock src/sol_execbench/core/data src/sol_execbench/core/scoring/amd_score.py src/sol_execbench/cli/main.py` was empty. |
| 9 | SANITY-01 through SANITY-04 are covered by CPU-safe tests and implementation evidence. | VERIFIED | Requirement-specific tests exist: `test_sanity_01_report_builds_existing_evidence_summary`, `test_sanity_02_diagnostic_statuses_do_not_recompute_score_eligibility`, `test_sanity_03_claim_boundaries_are_explicit_and_false`, `test_sanity_04_missing_optional_evidence_becomes_gap_without_probes`, plus script and public guardrail tests. Focused suite passed: 157 tests. |

**Score:** 9/9 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/sol_execbench/core/scoring/amd_bound_sanity.py` | Strict report models, source refs, rollups, JSON/checksum, Markdown, write helpers | VERIFIED | Exists, 798 lines, substantive strict Pydantic implementation. `gsd-sdk query verify.artifacts` passed. |
| `scripts/report_amd_bound_sanity.py` | Thin argparse wrapper from existing artifact paths | VERIFIED | Exists, 96 lines, delegates loading/build/write to core helpers. `--help` spot-check succeeded. |
| `tests/sol_execbench/test_amd_bound_sanity.py` | CPU-safe SANITY-01..04 core tests | VERIFIED | Exists, 463 lines, includes status, boundary, no-probe, deterministic output, and strict model tests. |
| `tests/sol_execbench/test_amd_bound_sanity_script.py` | CPU-safe script tests | VERIFIED | Exists, 196 lines, verifies deterministic JSON/Markdown and missing optional evidence gaps. |
| `tests/sol_execbench/test_public_contract_guardrails.py` | Public contract guardrails | VERIFIED | Exists and includes Phase 87 guardrails for sidecar-only fields, AMD score exclusion, primary CLI non-exposure, and negative boundary wording. |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `amd_bound_sanity.py` | `amd_sol_v2.py` | Consumes AMD SOL aggregate/coverage/warnings/hardware evidence | VERIFIED | `verify.key-links` found planned patterns; manual read confirms `aggregate_bound`, `coverage_summary`, warnings, and hardware model handling. |
| `amd_bound_sanity.py` | `solar_derivation.py` | Consumes SOLAR aggregate/coverage/warnings/source evidence | VERIFIED | `verify.key-links` found planned patterns; manual read confirms `aggregate_status`, `coverage_summary`, warnings, and source refs. |
| `amd_bound_sanity.py` | `amd_score.py` | Consumes AMD-native score support/evidence/warnings without changing semantics | VERIFIED | Builder copies `supported` to source evidence and diagnostic status only; guardrail tests protect score payloads. |
| `amd_bound_sanity.py` | `checksums.py` | Stable report checksum | VERIFIED | `stable_json_checksum` used in `AmdBoundSanityReport.with_checksum()`. |
| `report_amd_bound_sanity.py` | `amd_bound_sanity.py` | Delegates load/build/write | VERIFIED | Script imports `build_amd_bound_sanity_report`, `load_json`, and `write_amd_bound_sanity_reports`. |
| `test_public_contract_guardrails.py` | canonical data models | Sidecar-only contract checks | VERIFIED | Guardrails serialize canonical Definition/Workload/Trace payloads and reject bound sanity field leakage. |
| `test_public_contract_guardrails.py` | primary CLI | Non-exposure checks | VERIFIED | Guardrail invokes Click help and rejects bound sanity options/subcommands. |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|--------|
| `amd_bound_sanity.py` | `sources`, `artifact_availability`, `workloads`, `status_totals`, `evidence_gaps`, `claim_boundary` | Existing input dictionaries/refs passed to `build_amd_bound_sanity_report()` and explicit paths passed through script | Yes | VERIFIED |
| `scripts/report_amd_bound_sanity.py` | `report` | `load_json()` on explicit CLI paths, then core builder | Yes | VERIFIED |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Focused CPU-safe suite covers core, script, public guardrails, and related scoring regressions | `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_amd_bound_sanity.py tests/sol_execbench/test_amd_bound_sanity_script.py tests/sol_execbench/test_public_contract_guardrails.py tests/sol_execbench/test_amd_native_score.py tests/sol_execbench/test_amd_sol_bounds.py tests/sol_execbench/test_solar_derivation_evidence.py -q` | `157 passed in 8.02s` | PASS |
| Script is runnable without invoking benchmark/GPU/Docker paths | `UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/report_amd_bound_sanity.py --help` | Help displayed expected explicit sidecar-path options | PASS |
| No dependency/canonical/primary CLI tracked diffs | `git diff -- pyproject.toml uv.lock src/sol_execbench/core/data src/sol_execbench/core/scoring/amd_score.py src/sol_execbench/cli/main.py` | Empty diff | PASS |

### Probe Execution

| Probe | Command | Result | Status |
|-------|---------|--------|--------|
| GPU/ROCm/Docker probes | Not run | Explicitly skipped per phase/user scope; verification used CPU-safe grep, script help, and pytest only. | SKIPPED |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| SANITY-01 | 87-01, 87-02 | Emit `amd_bound_sanity.v1` over existing RDNA 4/Docker evidence with availability, statuses, coverage, warnings, gaps | SATISFIED | Core builder/report fields, script wrapper, and SANITY-01 tests verified. |
| SANITY-02 | 87-01, 87-02 | Distinguish scored/degraded/unscored/unsupported/provisional/missing-evidence without score eligibility changes | SATISFIED | Status vocabulary, rollup logic, `amd_score_supported` source evidence, AMD score guardrails. |
| SANITY-03 | 87-01, 87-02 | Surface provisional RDNA 4 model risk and avoid equivalence/validation/authority/hardware claims | SATISFIED | Claim boundary model/text and Markdown guardrails verify all required negative claims. |
| SANITY-04 | 87-01, 87-02 | Consume existing trace/closure/AMD SOL/SOLAR/score/compatibility refs/checksums without probes, Docker privilege changes, or relock | SATISFIED | Script loads explicit local JSON paths only; no `pyproject.toml`/`uv.lock` diff; no GPU/ROCm/Docker probes run or required. |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `tests/sol_execbench/test_amd_bound_sanity.py` | 375 | `subprocess.run` monkeypatch | INFO | Intentional no-probe guard. |
| `tests/sol_execbench/test_public_contract_guardrails.py` | multiple | ROCm/Docker strings | INFO | Existing guardrail fixtures and expected text, not executable probe behavior. |

### Human Verification Required

None.

### Gaps Summary

No blocking gaps found. Phase 87 achieves the roadmap goal and SANITY-01 through SANITY-04 while preserving the requested boundaries: no hardware-validation expansion, no GPU/ROCm/Docker probes, no dependency relock, no canonical schema changes, and no score eligibility changes.

---

_Verified: 2026-05-31T11:26:03Z_
_Verifier: the agent (gsd-verifier)_
