---
phase: 49
slug: high-confidence-family-modeling
status: draft
nyquist_compliant: true
wave_0_complete: false
created: 2026-05-23
---

# Phase 49 — Validation Strategy

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

- **After every task commit:** Run `uv run pytest tests/sol_execbench/test_solar_derivation_evidence.py -n 0 -x`.
- **After every plan wave:** Run `uv run pytest tests/sol_execbench/test_solar_derivation_evidence.py tests/sol_execbench/test_solar_derivation_contract.py tests/sol_execbench/test_public_contract_guardrails.py -n 0`.
- **Before `$gsd-verify-work`:** Run the phase gate plus `uv run pytest tests/sol_execbench/test_amd_bound_graph.py tests/sol_execbench/test_amd_sol_v2.py -n 0` and `uv run --with ruff ruff check src/sol_execbench/core/scoring/solar_derivation.py tests/sol_execbench/test_solar_derivation_evidence.py tests/sol_execbench/test_public_contract_guardrails.py`.
- **Max feedback latency:** 10 seconds for focused tests; 30 seconds for phase-gate regression tests.

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 49-01-01 | 01 | 1 | MODEL-01, MODEL-02 | T-49-01 | Formula/byte evidence parser rejects malformed or partial family evidence instead of silently accepting ambiguous data. | unit | `uv run pytest tests/sol_execbench/test_solar_derivation_evidence.py -n 0 -x` | ✅ | ⬜ pending |
| 49-02-01 | 02 | 2 | DERIVE-06, MODEL-05 | T-49-02 | Linear projection emits GEMM-compatible formula and SOL-bound evidence only when dimensions and dtype are explicit. | unit | `uv run pytest tests/sol_execbench/test_solar_derivation_evidence.py -n 0 -x` | ✅ | ⬜ pending |
| 49-03-01 | 03 | 3 | DERIVE-01, MODEL-01, MODEL-02, MODEL-05 | T-49-03 | Attention subroles degrade when axes, mask semantics, or required dimensions are incomplete. | unit | `uv run pytest tests/sol_execbench/test_solar_derivation_evidence.py -n 0 -x` | ✅ | ⬜ pending |
| 49-04-01 | 04 | 4 | DERIVE-03, DERIVE-05, MODEL-05 | T-49-04 | Convolution and memory-bound families emit dtype-aware byte evidence without changing public schemas or score eligibility. | guardrail | `uv run pytest tests/sol_execbench/test_solar_derivation_evidence.py tests/sol_execbench/test_public_contract_guardrails.py -n 0 -x` | ✅ | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [x] `tests/sol_execbench/test_solar_derivation_evidence.py` exists and already covers Phase 48 parse, serialize, confidence, source boundary, and fixture evidence behavior.
- [x] `tests/sol_execbench/test_public_contract_guardrails.py` exists and already guards canonical schemas, trace JSONL, primary CLI behavior, and AMD score eligibility.
- [x] Existing AMD bound graph and AMD SOL v2 regression tests are available for integration checks.
- [ ] Phase 49 implementation must extend focused evidence tests for every newly promoted family and every new machine-verifiable formula/byte field.

---

## Manual-Only Verifications

No ROCm hardware, paper-scale dataset run, MI300X/CDNA validation, or submitted
candidate execution is required for Phase 49. This phase validates
deterministic formula-backed evidence from canonical structure only.

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies.
- [x] Sampling continuity: no 3 consecutive tasks without automated verify.
- [x] Wave 0 covers all existing references.
- [x] No watch-mode flags.
- [x] Feedback latency < 30 seconds for phase-gate checks.
- [x] `nyquist_compliant: true` set in frontmatter.

**Approval:** approved 2026-05-23
