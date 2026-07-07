# Dataset Solutions Refactor Design

## Purpose

Reduce coupling in `src/sol_execbench/core/dataset/runner.py` by moving solution
construction and source sanitizing helpers into a focused module.

This repository does not currently need backward compatibility for internal
imports, so callers should import the new module directly instead of keeping
`runner.py` as a re-export surface.

## Scope

Create `src/sol_execbench/core/dataset/solutions.py`.

Move these names from `runner.py` into the new module:

- `infer_destination_passing_style`
- `sanitize_python_source_for_static_review`
- `build_solution_for_problem`
- `build_custom_solution`
- `build_reference_solution`

Remove these functions from `runner.py`; do not keep runner-level compatibility
aliases or re-exports.

## Callers

Update internal callers to import from `sol_execbench.core.dataset.solutions`
directly.

Expected caller updates:

- `scripts/run_dataset.py` imports `build_solution_for_problem`,
  `build_custom_solution`, and `build_reference_solution` from the new module.
- `scripts/internal/rdna4/run_rdna4_profiler_timing_batch.py` imports
  `build_reference_solution` from the new module.
- `tests/sol_execbench/test_dataset_runner.py` uses the new module for solution
  sanitizer and solution-construction tests.

## Architecture

`solutions.py` owns the conversion from SOL ExecBench definitions and local
solution files into benchmark Solution dictionaries. It also owns the token-aware
source sanitizer and destination-passing-style inference needed by that
conversion.

`runner.py` remains responsible for timing evidence, summary reporting, and
derived report orchestration.

The dependency direction should be:

- Callers import `solutions.py` directly.
- `solutions.py` does not import `runner.py`.
- `runner.py` should not import `solutions.py` unless a direct local use remains
  after extraction.

## Testing

Use test-first implementation.

Add or update focused coverage to prove:

- The token-aware stream sanitizer still rewrites exact `stream` identifiers
  without mutating comments, strings, or words such as `mainstream`.
- Reference solution construction still sanitizes source, sets metadata, and
  detects destination-passing style.
- Custom solution construction still preserves metadata and detects
  destination-passing style.
- Script-level dataset execution tests continue to pass after import updates.

Run the relevant regression set:

- `uv run pytest tests/sol_execbench/test_dataset_runner.py`
- `uv run pytest tests/sol_execbench/test_run_dataset_execution_closure.py`
- `uv run pytest tests/sol_execbench/test_run_dataset_amd_score.py`
- `uv run --with ruff ruff check` on changed files

## Non-Goals

Do not change Solution dictionary schema, metadata values, source paths,
destination-passing-style inference semantics, or stream sanitizing behavior.

Do not refactor CLI execution, AMD score report logic, timing evidence
collection, dataset execution closure logic, or summary reporting in this slice.
