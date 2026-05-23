---
phase: 57
slug: claim-guardrails-docs-and-release-closure
status: draft
nyquist_compliant: true
created: 2026-05-23
---

# Phase 57 - Validation Strategy

| Command | Purpose |
|---------|---------|
| `uv run pytest tests/sol_execbench/test_public_contract_guardrails.py -n 0 -x` | Claim wording and public contract guardrails |
| `uv run pytest tests/sol_execbench/test_parity_gap_report.py tests/sol_execbench/test_run_dataset_execution_closure.py tests/sol_execbench/test_dataset_inventory_readiness.py tests/sol_execbench/test_public_contract_guardrails.py -n 0` | Relevant v1.11 suite |
| `uv run --with ruff ruff check docs tests/sol_execbench/test_public_contract_guardrails.py` | Lint touched docs/tests where applicable |

Manual full-suite hardware validation remains deferred.
