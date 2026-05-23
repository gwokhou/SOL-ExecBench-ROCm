---
phase: 47
slug: derivation-contract-and-golden-fixture-matrix
status: draft
nyquist_compliant: true
wave_0_complete: true
created: 2026-05-23
---

# Phase 47 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 9.x |
| **Config file** | `pyproject.toml` |
| **Quick run command** | `uv run pytest tests/sol_execbench/test_solar_derivation_contract.py -n 0` |
| **Full suite command** | `uv run pytest tests/sol_execbench/test_solar_derivation_contract.py tests/sol_execbench/test_public_contract_guardrails.py -n 0` |
| **Estimated runtime** | ~20 seconds |

---

## Sampling Rate

- **After every task commit:** Run `uv run pytest tests/sol_execbench/test_solar_derivation_contract.py -n 0`
- **After every plan wave:** Run `uv run pytest tests/sol_execbench/test_solar_derivation_contract.py tests/sol_execbench/test_public_contract_guardrails.py -n 0`
- **Before `$gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** 30 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 47-01-01 | 01 | 1 | TEST-01, TEST-02 | — | N/A | docs | `test -f docs/internal/solar_derivation_contract.md` | ✅ | ⬜ pending |
| 47-02-01 | 02 | 1 | TEST-01, TEST-02 | — | N/A | fixture | `uv run pytest tests/sol_execbench/test_solar_derivation_contract.py -n 0` | ✅ | ⬜ pending |
| 47-03-01 | 03 | 2 | TEST-01, TEST-02 | — | N/A | guardrail | `uv run pytest tests/sol_execbench/test_solar_derivation_contract.py tests/sol_execbench/test_public_contract_guardrails.py -n 0` | ✅ | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

Existing infrastructure covers all phase requirements.

---

## Manual-Only Verifications

All phase behaviors have automated verification.

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all MISSING references
- [x] No watch-mode flags
- [x] Feedback latency < 30s
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** approved 2026-05-23
