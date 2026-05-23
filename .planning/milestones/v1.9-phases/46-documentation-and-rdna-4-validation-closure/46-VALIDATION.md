---
phase: 46
slug: documentation-and-rdna-4-validation-closure
status: draft
nyquist_compliant: true
wave_0_complete: true
created: 2026-05-23
---

# Phase 46 - Validation Strategy

## Commands

- `uv run pytest tests/sol_execbench/test_v1_9_validation_closure.py tests/sol_execbench/test_amd_native_score.py tests/sol_execbench/test_public_contract_guardrails.py -x`
- `uv run --with ruff ruff check tests/sol_execbench/test_v1_9_validation_closure.py tests/sol_execbench/test_amd_native_score.py`
