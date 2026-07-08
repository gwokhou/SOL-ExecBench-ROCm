# Quick Task Summary

Status: Implemented on branch `module-aligned-layout`.

Completed:

- Reorganized CLI modules under `src/sol_execbench/cli/commands/`,
  `src/sol_execbench/cli/evaluation/`, and `src/sol_execbench/cli/sidecars/`.
- Reorganized core report modules under `src/sol_execbench/core/reports/`.
- Reorganized platform modules under `src/sol_execbench/core/platform/`.
- Reorganized evidence modules under `src/sol_execbench/core/evidence/`.
- Moved tests to mirror the new source layout where practical:
  `tests/sol_execbench/cli/`, `tests/sol_execbench/core/reports/`,
  `tests/sol_execbench/core/platform/`, `tests/sol_execbench/core/evidence/`,
  `tests/sol_execbench/core/dataset/`, and
  `tests/sol_execbench/core/scoring/`.
- Kept root-level tests for cross-cutting contract, e2e, and derived-run
  isolation coverage.
- Preserved legacy public import paths with compatibility facades.
- Updated provenance, residue-audit, and boundary tests for canonical module
  paths.

Verification:

- `uv run --with ruff ruff check .`: passed.
- CLI/reports/provenance focused test batch: passed, 176 tests.
- Bench/dataset/scoring focused test batch: passed, 1098 tests, 57 skipped.
- Evidence public contract guardrails: passed, 51 tests.
- Import smoke for canonical and legacy facades: passed.
- `uv run pytest tests/sol_execbench -q --maxfail=20`: 1827 passed, 60 skipped,
  14 failed.

Known remaining failures:

- One existing CDNA3 schema audit failure expecting `gfx940`, `gfx941`, and
  `gfx942` in `src/sol_execbench/core/data/solution.py`.
- Docker wrapper/preflight failures in the local macOS shell environment,
  including `local: -n: invalid option`, unset array variables, missing sidecar
  output files, and mixed-version status mismatches.
