---
phase: 50
slug: degraded-complex-family-modeling
status: draft
nyquist_compliant: true
wave_0_complete: false
created: 2026-05-23
---

# Phase 50 — Validation Strategy

> Per-phase validation contract for conservative MoE and SSM/Mamba derivation.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 9.0.2 |
| **Config file** | `pyproject.toml` |
| **Quick run command** | `uv run pytest tests/sol_execbench/test_solar_derivation_family_modeling.py -n 0 -x` |
| **Full suite command** | `uv run pytest tests/sol_execbench/test_solar_derivation_family_modeling.py tests/sol_execbench/test_solar_derivation_evidence.py tests/sol_execbench/test_amd_bound_graph.py tests/sol_execbench/test_amd_bound_estimates.py tests/sol_execbench/test_public_contract_guardrails.py tests/sol_execbench/test_amd_sol_v2.py -n 0` |
| **Estimated runtime** | ~5 seconds |

---

## Sampling Rate

- **After every task commit:** Run the quick command or a narrower `-k "moe or ssm or mamba"` selector.
- **After every plan wave:** Run the full suite command.
- **Before `$gsd-verify-work`:** Run the full suite plus Ruff over touched scoring/test files.
- **Max feedback latency:** 10 seconds for focused tests; 30 seconds for phase-gate regression tests.

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 50-01-01 | 01 | 1 | DERIVE-02 | T-50-01 | MoE routing/top-k/dispatch/combine evidence degrades when static routing cardinality is incomplete. | unit | `uv run pytest tests/sol_execbench/test_solar_derivation_family_modeling.py tests/sol_execbench/test_amd_bound_graph.py tests/sol_execbench/test_amd_bound_estimates.py -k "moe" -n 0 -x` | ✅ | ⬜ pending |
| 50-02-01 | 02 | 2 | DERIVE-04 | T-50-02 | SSM/Mamba projection/depthwise-conv/scan/gating/output evidence degrades when recurrence or state semantics are incomplete. | unit | `uv run pytest tests/sol_execbench/test_solar_derivation_family_modeling.py tests/sol_execbench/test_amd_bound_graph.py tests/sol_execbench/test_amd_bound_estimates.py -k "ssm or mamba" -n 0 -x` | ✅ | ⬜ pending |
| 50-03-01 | 03 | 3 | DERIVE-02, DERIVE-04 | T-50-03 | Complex-family evidence stays sidecar-only, preserves Phase 49 high-confidence behavior, and leaves score eligibility unchanged. | guardrail | `uv run pytest tests/sol_execbench/test_solar_derivation_family_modeling.py tests/sol_execbench/test_public_contract_guardrails.py tests/sol_execbench/test_amd_sol_v2.py -n 0 -x` | ✅ | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [x] Phase 49 group-local formula, byte, and bound evidence tests exist.
- [x] Phase 49 high-confidence family tests exist and must remain green.
- [x] Public contract guardrails exist for canonical schemas, CLI, trace JSONL, and score eligibility.
- [ ] Phase 50 implementation must add MoE and SSM/Mamba positive/degraded/unsupported tests.

---

## Manual-Only Verifications

No ROCm hardware, paper-scale dataset run, hosted service, or candidate
solution execution is required for Phase 50. Verification is deterministic
parser, graph, estimate, sidecar, and guardrail coverage only.

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies.
- [x] Sampling continuity: no 3 consecutive tasks without automated verify.
- [x] Wave 0 covers existing references.
- [x] No watch-mode flags.
- [x] Feedback latency < 30 seconds for phase-gate checks.
- [x] `nyquist_compliant: true` set in frontmatter.

**Approval:** approved 2026-05-23
