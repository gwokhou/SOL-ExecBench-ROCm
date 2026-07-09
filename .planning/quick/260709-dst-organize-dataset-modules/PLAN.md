---
status: complete
created: 2026-07-09
---

# Organize Dataset Modules

## Goal

Reduce the remaining flat `src/sol_execbench/core/dataset/` module layout by
moving clear prefix-based feature clusters into subpackages.

## Scope

- Move `inventory*` modules into `dataset/inventory/`.
- Move `readiness*` modules into `dataset/readiness/`.
- Move `execution_closure*` modules into `dataset/execution_closure/`.
- Move `migration*` modules into `dataset/migration/`.
- Move `paper_denominator*` modules into `dataset/paper_denominator/`.
- Move `parity_gap*` modules into `dataset/parity_gap/`.
- Move `profiler_timing_coverage*` modules into
  `dataset/profiler_timing_coverage/`.
- Update imports, package exports, tests, and scripts.

## Out of Scope

- Runner and CLI execution module restructuring.
- Behavioral changes.
- Backward compatibility facade modules for old flat import paths.

## Verification

- Focused dataset/core tests.
- Ruff check.
- Ty check.
