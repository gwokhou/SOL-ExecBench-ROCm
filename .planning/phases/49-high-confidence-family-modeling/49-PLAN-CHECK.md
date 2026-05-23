# Phase 49 Plan Check: High-Confidence Family Modeling

**Checked:** 2026-05-23  
**Verdict:** PASS WITH WARNINGS  
**Plans checked:** 4  
**Blockers:** 0  
**Warnings:** 2  

## Goal-Backward Summary

Phase goal: users can derive formula-backed SOLAR evidence for high-confidence families whose dimensions and memory behavior are visible from reference or workload structure.

The plans are sufficient to deliver the phase goal. They first create the internal group-local formula, byte, and bound evidence contract, then populate it for linear projection, explicit attention, convolution, and embedding/positional/gather/rotary-like memory-bound families. The dependency chain is valid and matches the data flow needed by later plans.

## Requirement Coverage

| Requirement | Covering plan/tasks | Status |
|-------------|---------------------|--------|
| DERIVE-01 | 49-03 tasks 49-03-01, 49-03-02 | PASS |
| DERIVE-03 | 49-04 task 49-04-01 | PASS |
| DERIVE-05 | 49-04 task 49-04-02 | PASS |
| DERIVE-06 | 49-02 tasks 49-02-01, 49-02-02 | PASS |
| MODEL-01 | 49-01 contract plus 49-02/03/04 family population | PASS |
| MODEL-02 | 49-01 contract plus 49-02/03/04 family population | PASS |
| MODEL-05 | 49-01 bound evidence plus 49-02/03/04 family population | PASS |

## Decision Compliance

| Decision / boundary | Assessment |
|---------------------|------------|
| Linear projection uses `formula_kind="gemm_flops"` while preserving `op_family="linear_projection"` | PASS: 49-02 explicitly locks this and forbids `linear_projection_flops`. |
| Per-op formula, byte, and bound evidence lives inside semantic groups | PASS: 49-01 adds group-local evidence fields and 49-02/03/04 populate them. |
| Formula and byte evidence stays family-specific and dtype-aware | PASS: every family plan names formula inputs and read/write/intermediate/movement/total byte buckets. |
| Public schema, CLI, trace JSONL, candidate execution, and score eligibility boundaries are preserved | PASS WITH WARNING: the intended boundary is correct, but guardrail wording should be narrowed as described below. |
| Deferred Phase 50/51/52 work remains out of scope | PASS: MoE/SSM, score eligibility integration, reporting, docs, dataset-scale extraction, and hardware validation are excluded or guarded. |

## Structural Checks

| Dimension | Status | Notes |
|-----------|--------|-------|
| Task completeness | PASS | `gsd-sdk query verify.plan-structure` reports all tasks valid with files/action/verify/done. |
| Dependency correctness | PASS | 49-01 -> 49-02 -> 49-03 -> 49-04 is acyclic and matches the shared contract before population. |
| Key links planned | PASS | Plans wire graph attributes -> estimates -> `SolarSemanticGroupEvidence` -> parser/guardrails. |
| Scope sanity | PASS | Plans have 2-3 tasks and 3-7 modified files each; no 5-task or 15-file plan. |
| must_haves derivation | PASS | Truths are user-observable and map to artifacts/key links. |
| Architectural tier compliance | PASS | Recognition/modeling remains in scoring/derivation backend; public layers stay guardrail-only. |
| Cross-plan data contracts | PASS | `OperatorWorkEstimate` remains numeric source of truth; group-local evidence is populated from it. |
| AGENTS.md compliance | WARNING | Plan expected commit examples use conventional commit style, not the repository `#<Issue Number> - <Title>` plus DCO convention. |
| Research resolution | PASS | `49-RESEARCH.md` has resolved planning decisions and no unresolved open questions section. |
| Pattern compliance | PASS | Plans reference and follow the PATTERNS.md analogs for sidecar dataclasses, strict parsers, estimate helpers, graph classifiers, and guardrails. |

## Nyquist Compliance

`49-VALIDATION.md` exists and `nyquist_compliant: true`.

| Task | Plan | Wave | Automated command | Status |
|------|------|------|-------------------|--------|
| 49-01-01 | 49-01 | 1 | `uv run pytest tests/sol_execbench/test_solar_derivation_evidence.py -k "formula or byte or bound or round_trip" -n 0 -x` | PASS |
| 49-01-02 | 49-01 | 1 | `uv run pytest tests/sol_execbench/test_solar_derivation_evidence.py -k "parser or deterministic or formula or byte or bound" -n 0 -x` | PASS |
| 49-01-03 | 49-01 | 1 | `uv run pytest tests/sol_execbench/test_public_contract_guardrails.py tests/sol_execbench/test_solar_derivation_evidence.py -n 0 -x` | PASS |
| 49-02-01 | 49-02 | 2 | `uv run pytest tests/sol_execbench/test_amd_bound_estimates.py -k "linear or projection or gemm" -n 0 -x` | PASS |
| 49-02-02 | 49-02 | 2 | `uv run pytest tests/sol_execbench/test_solar_derivation_evidence.py -k "linear or projection or formula or byte or bound" -n 0 -x` | PASS |
| 49-03-01 | 49-03 | 3 | `uv run pytest tests/sol_execbench/test_amd_bound_graph.py tests/sol_execbench/test_solar_derivation_family_modeling.py -k attention -n 0 -x` | PASS |
| 49-03-02 | 49-03 | 3 | `uv run pytest tests/sol_execbench/test_amd_bound_estimates.py tests/sol_execbench/test_solar_derivation_family_modeling.py -k "attention and (formula or byte or bound or degraded)" -n 0 -x` | PASS |
| 49-04-01 | 49-04 | 4 | `uv run pytest tests/sol_execbench/test_amd_bound_graph.py tests/sol_execbench/test_amd_bound_estimates.py tests/sol_execbench/test_solar_derivation_family_modeling.py -k "conv or convolution" -n 0 -x` | PASS |
| 49-04-02 | 49-04 | 4 | `uv run pytest tests/sol_execbench/test_amd_bound_graph.py tests/sol_execbench/test_amd_bound_estimates.py tests/sol_execbench/test_solar_derivation_family_modeling.py -k "embedding or positional or gather or rotary" -n 0 -x` | PASS |
| 49-04-03 | 49-04 | 4 | `uv run pytest tests/sol_execbench/test_solar_derivation_family_modeling.py tests/sol_execbench/test_solar_derivation_evidence.py tests/sol_execbench/test_amd_bound_graph.py tests/sol_execbench/test_amd_bound_estimates.py tests/sol_execbench/test_public_contract_guardrails.py tests/sol_execbench/test_amd_sol_v2.py -n 0` | PASS |

Sampling: every implementation task has an automated verification command; no watch-mode flags; final phase gate includes family tests, parser tests, public guardrails, graph/estimate tests, and AMD SOL v2 regression.

## Findings

### Warnings

**1. [public_boundary] Guardrail wording may forbid existing AMD SOL bound fields**
- Plan: 49-01 and 49-04
- Severity: WARNING
- Detail: Plan 49-01 lists `compute_bound_ms`, `memory_bound_ms`, `sol_bound_ms`, and `limiting_resource` as sidecar-only field names to keep out of public surfaces. These fields already exist in AMD SOL v1/v2 bound artifacts and tests (`AmdSolV2OpBound`, `AmdSolBound`), so a literal forbidden-field assertion against AMD-native artifacts would either fail or force an unintended public artifact change.
- Fix: Scope the guardrail to canonical `Definition`, `Workload`, `Trace`, primary CLI help, `evidence_refs`, and new SOLAR-only names such as `formula_evidence`, `byte_evidence`, and `bound_evidence`. Do not forbid existing AMD SOL bound fields in AMD SOL artifacts.

**2. [agents_compliance] Expected commit examples do not match repository commit convention**
- Plan: 49-01 through 49-04
- Severity: WARNING
- Detail: The plans list expected commits such as `feat(49-04): derive convolution and memory-bound evidence`, while AGENTS.md asks for commit titles in the form `#<Issue Number> - <Commit Title>` and DCO sign-off.
- Fix: During execution, use the repository commit format and `git commit -s`; the plan examples should be treated as topical hints, not literal commit titles.

## Structured Issues

```yaml
issues:
  - plan: "49-01"
    dimension: "public_boundary"
    severity: "warning"
    description: "Guardrail wording may forbid existing AMD SOL bound artifact fields such as compute_bound_ms, memory_bound_ms, sol_bound_ms, and limiting_resource."
    task: "49-01-03"
    fix_hint: "Only forbid new SOLAR derivation sidecar fields from canonical schemas, primary CLI help, AMD score evidence_refs, and inappropriate public surfaces; do not forbid existing AMD SOL v1/v2 bound fields in AMD SOL artifacts."
  - plan: "49-04"
    dimension: "public_boundary"
    severity: "warning"
    description: "End-of-phase guardrails should preserve existing AMD-native artifact schema while preventing SOLAR sidecar leakage."
    task: "49-04-03"
    fix_hint: "Keep AMD-native score eligibility unchanged and assert absence of solar_derivation/evidence_refs wiring, not absence of existing bound fields."
  - plan: null
    dimension: "agents_compliance"
    severity: "warning"
    description: "Expected commit examples use conventional commit style instead of the repository AGENTS.md issue-number and DCO convention."
    task: null
    fix_hint: "Use DCO-signed commits and the repository commit title format during execution."
```

## Recommendation

Execution can proceed after acknowledging the warnings. No blocker requires replanning, but the executor should narrow public guardrail assertions before implementing tests so existing AMD SOL v1/v2 artifact fields are preserved.
