---
status: complete
created: 2026-07-09
---

# Organize Report Modules

## Goal

Reduce flat `src/sol_execbench/core/reports/` module layout by moving clear
prefix-based report clusters into subpackages.

## Scope

- Move `consistency*` modules into `reports/consistency/`.
- Move `matrix_diff*` modules into `reports/matrix_diff/`.
- Move `trust_summary*` modules into `reports/trust_summary/`.
- Move `claim_upgrade*` modules into `reports/claim_upgrade/`.
- Move `evaluation_stability*` modules into `reports/evaluation_stability/`.
- Update imports, tests, docs, and boundary allowlists.

## Out of Scope

- Shared reporting utilities such as `reporting.py`, `report_payloads.py`.
- Behavioral changes.
- Backward compatibility facade modules for old flat import paths.

## Verification

- Focused report tests.
- Ruff check.
- Ty check.
- Full pytest if focused validation passes.
