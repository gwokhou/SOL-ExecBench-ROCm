---
phase: 56
slug: parity-gap-reporting-and-evidence-review
status: verified
verified: 2026-05-23
---

# Phase 56 Verification

## Automated Checks

| Command | Result |
|---------|--------|
| `uv run pytest tests/sol_execbench/test_parity_gap_report.py -n 0 -x` | pass, 4 tests |
| `uv run pytest tests/sol_execbench/test_parity_gap_report.py tests/sol_execbench/test_run_dataset_execution_closure.py tests/sol_execbench/test_dataset_inventory_readiness.py tests/sol_execbench/test_public_contract_guardrails.py -n 0` | pass, 51 tests |
| `uv run --with ruff ruff check src/sol_execbench/core/dataset/parity_gap.py src/sol_execbench/core/dataset/__init__.py scripts/report_parity_gaps.py tests/sol_execbench/test_parity_gap_report.py tests/sol_execbench/test_public_contract_guardrails.py` | pass |

## Requirement Coverage

- GAP-01: JSON and Markdown parity-gap reports are generated from sidecars.
- GAP-02: Suite/category denominators cover discovered, parsed, ready, blocked,
  not attempted, skipped, attempted, passed, failed, scored, degraded, and
  unscored.
- GAP-03: Blockers are grouped by stable reason code with next actions.
- GAP-04: Evidence completeness distinguishes trace, timing, AMD-native score,
  AMD SOL, and SOLAR derivation evidence.
- GAP-05: Report source refs and artifact refs are deterministic sidecar fields.

## Manual Limits

The report summarizes available sidecar evidence. It does not run full-suite
ROCm validation or establish paper parity.
