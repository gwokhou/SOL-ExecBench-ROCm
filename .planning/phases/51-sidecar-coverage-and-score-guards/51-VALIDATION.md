---
phase: 51
slug: sidecar-coverage-and-score-guards
status: draft
nyquist_compliant: true
wave_0_complete: false
created: 2026-05-23
---

# Phase 51 — Validation Strategy

> Per-phase validation contract for SOLAR sidecar coverage and score guards.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 9.0.2 |
| **Config file** | `pyproject.toml` |
| **Quick run command** | `uv run pytest tests/sol_execbench/test_solar_derivation_evidence.py -n 0 -x` |
| **Full suite command** | `uv run pytest tests/sol_execbench/test_solar_derivation_family_modeling.py tests/sol_execbench/test_solar_derivation_evidence.py tests/sol_execbench/test_amd_bound_graph.py tests/sol_execbench/test_amd_bound_estimates.py tests/sol_execbench/test_public_contract_guardrails.py tests/sol_execbench/test_amd_sol_v2.py -n 0` |
| **Estimated runtime** | ~5 seconds |

---

## Sampling Rate

- **After every task commit:** Run focused parser/coverage or score guard tests.
- **After every plan wave:** Run the full suite command.
- **Before verification:** Run the full suite plus Ruff over touched scoring/test files.
- **Max feedback latency:** 10 seconds for focused tests; 30 seconds for phase-gate regression tests.

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 51-01-01 | 01 | 1 | REPORT-01, REPORT-02, TEST-03 | T-51-01 | Coverage sidecars parse strictly and expose scored/degraded/unscored states without public schema drift. | unit | `uv run pytest tests/sol_execbench/test_solar_derivation_evidence.py -k "coverage or aggregate or round_trip" -n 0 -x` | ✅ | ⬜ pending |
| 51-02-01 | 02 | 2 | REPORT-03 | T-51-02 | AMD-native score guards return `None` for unscored SOLAR evidence and preserve degraded warnings. | unit | `uv run pytest tests/sol_execbench/test_amd_sol_v2.py tests/sol_execbench/test_public_contract_guardrails.py -k "score or solar or degraded or unscored" -n 0 -x` | ✅ | ⬜ pending |
| 51-03-01 | 03 | 3 | REPORT-01, REPORT-02, REPORT-03, TEST-03 | T-51-03 | Full phase gate preserves Phase 49/50 modeling and public boundaries. | guardrail | `uv run pytest tests/sol_execbench/test_solar_derivation_family_modeling.py tests/sol_execbench/test_solar_derivation_evidence.py tests/sol_execbench/test_amd_bound_graph.py tests/sol_execbench/test_amd_bound_estimates.py tests/sol_execbench/test_public_contract_guardrails.py tests/sol_execbench/test_amd_sol_v2.py -n 0` | ✅ | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [x] Phase 48-50 sidecar parser and family modeling tests exist.
- [x] Public contract guardrails exist.
- [x] AMD SOL v2 score/artifact tests exist.
- [ ] Phase 51 must add round-trip coverage for all new machine-verifiable coverage/score guard fields.

---

## Manual-Only Verifications

No ROCm hardware, paper-scale dataset run, hosted service, or candidate
solution execution is required for Phase 51. Verification is deterministic
parser, sidecar, score guard, and public-boundary coverage.

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies.
- [x] Sampling continuity: no 3 consecutive tasks without automated verify.
- [x] Wave 0 covers existing references.
- [x] No watch-mode flags.
- [x] Feedback latency < 30 seconds for phase-gate checks.
- [x] `nyquist_compliant: true` set in frontmatter.

**Approval:** approved 2026-05-23
