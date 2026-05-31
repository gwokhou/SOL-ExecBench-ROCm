---
nyquist_compliant: true
wave_0_complete: true
validated: 2026-05-31
---

# Phase 90 Nyquist Validation

Phase 90 has tests for representative stable and unstable timing states,
script generation, ROCm-shaped timing evidence, deterministic output, and
sidecar-only public contract boundaries.

| Task | Status | Evidence |
| --- | --- | --- |
| Classification matrix | green | `tests/sol_execbench/test_evaluation_stability.py` |
| ROCm-shaped timing evidence | green | `Rocprofv3TimingEvidence` regression |
| Script validation | green | `tests/sol_execbench/test_evaluation_stability_script.py` |
| Public contract boundary | green | `tests/sol_execbench/test_public_contract_guardrails.py` |

