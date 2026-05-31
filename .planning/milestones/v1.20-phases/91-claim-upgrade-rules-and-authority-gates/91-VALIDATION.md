---
nyquist_compliant: true
wave_0_complete: true
validated: 2026-05-31
---

# Phase 91 Nyquist Validation

Claim-upgrade has tests for eligible and blocked paths, missing evidence,
AMD SOL/SOLAR wiring, source refs, script generation, deterministic output, and
public authority guardrails.

| Task | Status | Evidence |
| --- | --- | --- |
| Rule coverage | green | `tests/sol_execbench/test_claim_upgrade.py` |
| Script validation | green | `tests/sol_execbench/test_claim_upgrade_script.py` |
| Cross-script wiring | green | `tests/sol_execbench/test_v1_20_evidence_quality_e2e.py` |
| Public contract boundary | green | `tests/sol_execbench/test_public_contract_guardrails.py` |

