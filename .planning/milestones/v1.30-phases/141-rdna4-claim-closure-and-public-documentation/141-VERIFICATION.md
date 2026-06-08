---
phase: 141
title: RDNA4 claim closure and public documentation
status: verified
verified_at: 2026-06-08
---

# Phase 141 Verification

## Commands

- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_public_contract_guardrails.py -q`
- `UV_CACHE_DIR=/tmp/uv-cache uv run --with ruff ruff check tests/sol_execbench/test_public_contract_guardrails.py`

## Results

- Public-contract guardrails passed: 51 passed.
- Ruff passed for the updated guardrail test file.

## Residual Risk

- Public docs intentionally preserve Phase 138 failures, 12 missing traces, 56
  temporary sidecar exclusions, and Phase 139 non-authoritative timing blockers.
- This phase closes wording and guardrails only; it does not rerun RDNA4 timing,
  resolve GPU clock sudoers gaps, or expand CDNA3/CDNA4 validation.

