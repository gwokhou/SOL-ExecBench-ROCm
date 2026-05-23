---
phase: 48
slug: extraction-pipeline-and-semantic-provenance
status: draft
nyquist_compliant: true
wave_0_complete: false
created: 2026-05-23
---

# Phase 48 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 9.0.2 |
| **Config file** | `pyproject.toml` |
| **Quick run command** | `uv run pytest tests/sol_execbench/test_solar_derivation_evidence.py -n 0 -x` |
| **Full suite command** | `uv run pytest tests/sol_execbench/test_solar_derivation_evidence.py tests/sol_execbench/test_solar_derivation_contract.py tests/sol_execbench/test_public_contract_guardrails.py -n 0` |
| **Estimated runtime** | ~5 seconds |

---

## Sampling Rate

- **After every task commit:** Run `uv run pytest tests/sol_execbench/test_solar_derivation_evidence.py -n 0 -x`
- **After every plan wave:** Run `uv run pytest tests/sol_execbench/test_solar_derivation_evidence.py tests/sol_execbench/test_solar_derivation_contract.py tests/sol_execbench/test_public_contract_guardrails.py -n 0`
- **Before `$gsd-verify-work`:** Run the full suite plus `uv run pytest tests/sol_execbench/test_amd_bound_graph.py tests/sol_execbench/test_amd_sol_v2.py -n 0`
- **Max feedback latency:** 10 seconds for focused tests; 30 seconds for phase-gate regression tests.

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 48-01-01 | 01 | 1 | DERIVE-07, MODEL-03 | T-48-01 | Strict sidecar parser rejects malformed evidence payloads. | unit | `uv run pytest tests/sol_execbench/test_solar_derivation_evidence.py -n 0 -x` | ❌ W0 | ⬜ pending |
| 48-02-01 | 02 | 1 | DERIVE-07, MODEL-03 | T-48-02 | Builder derives only from definition/workload/graph inputs, not candidate solutions. | unit | `uv run pytest tests/sol_execbench/test_solar_derivation_evidence.py -n 0 -x` | ❌ W0 | ⬜ pending |
| 48-03-01 | 03 | 2 | MODEL-04 | T-48-03 | Ambiguous evidence degrades to inexact or unsupported with missing evidence. | unit | `uv run pytest tests/sol_execbench/test_solar_derivation_evidence.py tests/sol_execbench/test_solar_derivation_contract.py -n 0 -x` | ❌ W0 | ⬜ pending |
| 48-04-01 | 04 | 2 | DERIVE-07, MODEL-03, MODEL-04 | T-48-04 | Public schemas, canonical traces, and primary CLI remain free of derivation evidence fields. | guardrail | `uv run pytest tests/sol_execbench/test_public_contract_guardrails.py -n 0 -x` | ✅ | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/sol_execbench/test_solar_derivation_evidence.py` — new focused tests for DERIVE-07, MODEL-03, and MODEL-04.
- [ ] Existing `tests/sol_execbench/test_public_contract_guardrails.py` covers public contract guardrails and must receive Phase 48 field-name assertions.
- [ ] Existing `tests/sol_execbench/test_solar_derivation_contract.py` covers Phase 47 fixture contract and should remain green.

---

## Manual-Only Verifications

All Phase 48 behaviors have automated verification. No ROCm hardware or manual
dataset run is required because this phase adds internal evidence contracts and
deterministic extraction plumbing only.

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies.
- [x] Sampling continuity: no 3 consecutive tasks without automated verify.
- [x] Wave 0 covers all missing references.
- [x] No watch-mode flags.
- [x] Feedback latency < 30 seconds for phase-gate checks.
- [x] `nyquist_compliant: true` set in frontmatter.

**Approval:** approved 2026-05-23
