---
phase: 48
verified: 2026-05-23T05:47:45Z
status: passed
score: 12/12 must-haves verified
overrides_applied: 0
automated_checks:
  - command: "uv run pytest tests/sol_execbench/test_solar_derivation_evidence.py tests/sol_execbench/test_solar_derivation_contract.py tests/sol_execbench/test_public_contract_guardrails.py -n 0"
    result: "58 passed in 1.20s"
  - command: "uv run pytest tests/sol_execbench/test_amd_bound_graph.py tests/sol_execbench/test_amd_sol_v2.py -n 0"
    result: "19 passed in 1.58s"
  - command: "uv run --with ruff ruff check src/sol_execbench/core/scoring/solar_derivation.py tests/sol_execbench/test_solar_derivation_evidence.py tests/sol_execbench/test_public_contract_guardrails.py"
    result: "passed"
---

# Phase 48: Extraction Pipeline And Semantic Provenance Verification Report

**Phase Goal:** The derivation pipeline can produce compound-family grouping, subrole, shape, dtype, axis, source, and confidence evidence without changing canonical benchmark artifacts.
**Verified:** 2026-05-23T05:47:45Z
**Status:** passed
**Re-verification:** No - initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Internal SOLAR evidence records compound-family grouping, subroles, and provenance outside canonical trace JSONL and public schemas. | VERIFIED | `SolarSemanticGroupEvidence`, `SolarSubroleEvidence`, `SolarDerivationEvidence`, and sidecar `source_boundary` exist in `src/sol_execbench/core/scoring/solar_derivation.py:84`, `:108`, `:151`, `:159`; public schema guardrail excludes Phase 48 fields at `tests/sol_execbench/test_public_contract_guardrails.py:161`. |
| 2 | Tensor shape, dtype, semantic-axis, and extraction-source provenance are inspectable for extracted evidence. | VERIFIED | `SolarTensorEvidence` stores `shape`, `dtype`, `semantic_axes`, `source`, and producer/missing evidence at `solar_derivation.py:58`; extraction populates these at `:501`; test coverage asserts shape/dtype/axes/source at `test_solar_derivation_evidence.py:501`. |
| 3 | Supported, inexact, and unsupported derivation states are deterministic. | VERIFIED | `classify_solar_confidence` maps metadata completeness to `supported/scored`, `inexact/degraded`, and `unsupported/unscored` at `solar_derivation.py:216`; tests cover all three states at `test_solar_derivation_evidence.py:722`, `:755`, `:783`. |
| 4 | Existing primary `sol-execbench` behavior and canonical benchmark schemas remain unchanged while sidecar evidence expands. | VERIFIED | Guardrails assert canonical `Definition`, `Workload`, and `Trace` dumps exclude Phase 48 fields and primary CLI help excludes Phase 48 options at `test_public_contract_guardrails.py:161` and `:203`; `rg` found no Phase 48 fields in `src/sol_execbench/core/data`, `src/sol_execbench/cli`, or existing score artifacts. |
| 5 | Parser and serializer round-trip evidence strictly and reject malformed payloads. | VERIFIED | `solar_derivation_from_dict` requires exact top-level/nested keys and validates schema, confidence, status, lists, booleans, and shape dimensions at `solar_derivation.py:299`; tests cover round-trip and malformed payload rejection at `test_solar_derivation_evidence.py:128`, `:157`, `:203`, `:217`, `:245`, `:260`, `:292`. |
| 6 | Evidence generation records source boundaries as sidecar-only and candidate-execution false. | VERIFIED | `_default_source_boundary()` returns all false at `solar_derivation.py:928`; parser preserves booleans at `:485`; tests assert boundary flags at `test_solar_derivation_evidence.py:260`, `:481`, `:589`. |
| 7 | Builder derives from `Definition`, `Workload`, `BoundGraph`, and `OperatorWorkEstimate` only. | VERIFIED | `build_solar_derivation_evidence(definition, workload)` calls `build_bound_graph(definition, workload)` and `estimate_bound_work(graph)` once each at `solar_derivation.py:176`; lower-level derive accepts only definition/workload/graph/estimates at `:186`. |
| 8 | Candidate solution code is not a derivation input. | VERIFIED | `solar_derivation.py` imports no `Solution`, driver, CLI, subprocess, or runner code; builder signatures are tested for no `Solution`, `solution_path`, `candidate`, or `submitted_code` parameters at `test_solar_derivation_evidence.py:589`. Existing FX extraction executes canonical `Definition.reference`, not submitted candidate solutions. |
| 9 | Semantic groups are deterministic and include family, node, tensor, subrole, source, confidence, status, missing-evidence, and warning metadata. | VERIFIED | `_semantic_group_evidence`, `_subroles_for_group`, and `SolarSemanticGroupEvidence.to_dict()` populate and sort group/subrole data at `solar_derivation.py:535`, `:654`, `:108`; deterministic serialization is tested at `test_solar_derivation_evidence.py:830`. |
| 10 | Ambiguous or incomplete evidence becomes degraded or unscored with explicit missing evidence and warnings. | VERIFIED | `classify_solar_confidence` emits missing evidence and stable warning prefixes at `solar_derivation.py:216`; tests assert degraded/unsupported missing evidence, warning prefixes, and non-scored status at `test_solar_derivation_evidence.py:755` and `:783`. |
| 11 | Phase 47 fixture expectations are representable as parseable Phase 48 evidence without executing fixture references or claiming paper/hardware validation. | VERIFIED | Fixture round-trip coverage uses `load_solar_derivation_fixtures()` and parses evidence at `test_solar_derivation_evidence.py:386`; boundary claims stay false at `:438`; fixture loader non-execution is covered in `test_solar_derivation_contract.py`. |
| 12 | AMD-native score eligibility and existing v1/v2 sidecar artifacts do not drift. | VERIFIED | Guardrail imports `solar_derivation` and verifies v1/v2 scores remain supported, claim level remains `amd-native-derived`, and `solar_derivation` is absent from score evidence refs and artifacts at `test_public_contract_guardrails.py:223`; AMD bound graph and v2 tests passed. |

**Score:** 12/12 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/sol_execbench/core/scoring/solar_derivation.py` | Internal evidence model, parser, builder, grouping, and confidence rules | VERIFIED | Exists and substantive. `gsd-sdk query verify.artifacts` passed for all four plans. Contains schema version, dataclasses, builder/derive helpers, strict parser, source boundaries, semantic grouping, and confidence classifier. |
| `tests/sol_execbench/test_solar_derivation_evidence.py` | Focused parser, builder, provenance, confidence, fixture, and boundary tests | VERIFIED | Exists and substantive. Contains required tests for round-trip, candidate boundary, tensor provenance, deterministic grouping, confidence states, and fixture representation. |
| `tests/sol_execbench/test_public_contract_guardrails.py` | Public schema, trace JSONL, primary CLI, and AMD score guardrails | VERIFIED | Exists and substantive. Contains forbidden field/option tests and AMD-native score eligibility guardrails for Phase 48 imports. |
| `tests/sol_execbench/test_solar_derivation_contract.py` | Phase 47 fixture contract and fixture non-execution baseline | VERIFIED | Existing contract remains green and supports Phase 48 fixture coverage. |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `SolarDerivationEvidence.to_dict` | `solar_derivation_from_dict` | JSON-safe dict round trip | VERIFIED | `test_solar_derivation_round_trip_preserves_provenance` asserts `solar_derivation_from_dict(artifact.to_dict()).to_dict() == artifact.to_dict()`. |
| `SolarTensorEvidence` | `MODEL-03` | Shape, dtype, semantic axes, source fields | VERIFIED | Fields are defined in the dataclass and populated by `_tensor_evidence`; tests assert shape/dtype/axes/source on derived evidence. |
| `build_solar_derivation_evidence` | `build_bound_graph` and `estimate_bound_work` | Definition/workload graph and estimate pipeline | VERIFIED | Direct calls are present at `solar_derivation.py:181` and `:182`; tests exercise builder output. |
| `derive_solar_derivation_evidence` | `SolarSemanticGroupEvidence` | BoundGraphNode family and estimate evidence | VERIFIED | `_semantic_group_evidence` builds groups from estimates and graph nodes; tests assert family, node, subrole, source, confidence, and status metadata. |
| Public guardrails | Canonical data models and primary CLI help | Forbidden Phase 48 fields and options | VERIFIED | `test_v1_10_solar_derivation_fields_remain_noncanonical` and `test_primary_cli_does_not_expose_v1_10_solar_derivation_options` passed. |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|--------|
| `SolarDerivationEvidence.groups` | `groups` | `_semantic_group_evidence(graph, estimates, nodes_by_id, tensor_evidence_by_id)` | Yes - groups are derived from `OperatorWorkEstimate` records and `BoundGraphNode`/tensor IDs, not static empty data. | VERIFIED |
| `SolarDerivationEvidence.tensors` | `tensors` | `_tensor_evidence(definition, workload, graph, tensor)` over `graph.tensors` | Yes - shape/dtype/source comes from `BoundTensor`; semantic axes come from definition/workload axis matching. | VERIFIED |
| `SolarConfidenceClassification` | `confidence`, `status`, `missing_evidence`, `warning_prefixes` | `classify_solar_confidence(...)` over family, nodes, tensors, estimates, and subroles | Yes - output is computed from metadata completeness and estimate confidence. | VERIFIED |
| Public guardrail assertions | canonical dumps and CLI help | `Definition.model_dump`, `Workload.model_dump`, `Trace.model_dump`, `CliRunner().invoke(cli, ["--help"])` | Yes - tests inspect current schema dumps and live Click help output. | VERIFIED |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Phase 48 evidence, contract, and public guardrails pass | `uv run pytest tests/sol_execbench/test_solar_derivation_evidence.py tests/sol_execbench/test_solar_derivation_contract.py tests/sol_execbench/test_public_contract_guardrails.py -n 0` | `58 passed in 1.20s` | PASS |
| Existing bound graph and v2 sidecar behavior remains intact | `uv run pytest tests/sol_execbench/test_amd_bound_graph.py tests/sol_execbench/test_amd_sol_v2.py -n 0` | `19 passed in 1.58s` | PASS |
| Ruff accepts touched Phase 48 files | `uv run --with ruff ruff check src/sol_execbench/core/scoring/solar_derivation.py tests/sol_execbench/test_solar_derivation_evidence.py tests/sol_execbench/test_public_contract_guardrails.py` | `All checks passed!` | PASS |
| No Phase 48 field leakage into canonical data/CLI/scoring surfaces | `rg -n "solar_derivation|semantic_axes|source_kind|source_detail|confidence_rationale|formula_provenance|byte_provenance|semantic_groups" src/sol_execbench/core/data src/sol_execbench/cli src/sol_execbench/core/scoring/amd_sol.py src/sol_execbench/core/scoring/amd_sol_v2.py src/sol_execbench/core/scoring/amd_score.py` | No matches | PASS |

### Probe Execution

No Phase 48 probe scripts were declared or discovered. Step 7c skipped.

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| DERIVE-07 | 48-01, 48-02, 48-03, 48-04 | Emits compound-family grouping, subrole, and provenance metadata without mutating canonical trace JSONL or public schemas. | SATISFIED | Semantic group/subrole/source records exist in `solar_derivation.py`; public schema/CLI guardrails passed; `rg` found no leakage into canonical surfaces. |
| MODEL-03 | 48-01, 48-02, 48-03, 48-04 | Formula and byte evidence carries tensor shape, dtype, semantic-axis, and extraction-source provenance. | SATISFIED | Tensor evidence includes shape/dtype/semantic axes/source; group required evidence includes formula/byte/axis provenance from estimates; tests assert derived tensor provenance and fixture representation. |
| MODEL-04 | 48-03, 48-04 | Deterministic supported, inexact, and unsupported confidence rules based on metadata completeness and recognized semantics. | SATISFIED | `classify_solar_confidence` implements deterministic rules and tests cover complete, incomplete, and unsupported groups. |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| None | n/a | n/a | n/a | `rg` found no `TBD`, `FIXME`, `XXX`, `TODO`, `HACK`, placeholder text, empty implementations, or console-only handlers in modified Phase 48 files. |

### Human Verification Required

None. Phase 48 is internal evidence and deterministic contract logic; the validation strategy states no ROCm hardware or manual dataset run is required.

### Gaps Summary

No blocking gaps found. Phase 48 achieves the goal: the derivation pipeline now produces internal sidecar evidence for grouping, subroles, tensor metadata, extraction sources, and deterministic confidence/status rules while preserving canonical benchmark artifacts, primary CLI behavior, and existing AMD-native score eligibility.

---

_Verified: 2026-05-23T05:47:45Z_
_Verifier: the agent (gsd-verifier)_
