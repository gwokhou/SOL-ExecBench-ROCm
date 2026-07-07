# Dataset Test Boundaries Design

## Purpose

Align dataset tests with the production module boundaries introduced by recent
refactors. `test_dataset_runner.py` still contains tests for `solutions.py` and
`cli_execution.py`, even though those modules now own their behavior.

## Scope

Create focused test files:

- `tests/sol_execbench/test_dataset_solutions.py`
- `tests/sol_execbench/test_cli_execution.py`

Move solution-construction tests from `test_dataset_runner.py` to
`test_dataset_solutions.py`.

Move CLI subprocess/logging tests from `test_dataset_runner.py` to
`test_cli_execution.py`.

Keep runner summary/reporting tests in `test_dataset_runner.py`.

## Architecture

The tests should mirror the current production boundaries:

- `test_dataset_solutions.py` imports `sol_execbench.core.dataset.solutions`.
- `test_cli_execution.py` imports `sol_execbench.core.dataset.cli_execution`.
- `test_dataset_runner.py` imports `sol_execbench.core.dataset.runner`.

No production code should change in this slice.

## Testing

Use test-preserving moves. The assertions should remain behavior-equivalent; only
module/file placement and imports should change.

Run:

- `uv run pytest tests/sol_execbench/test_dataset_runner.py`
- `uv run pytest tests/sol_execbench/test_dataset_solutions.py`
- `uv run pytest tests/sol_execbench/test_cli_execution.py`
- `uv run pytest tests/sol_execbench/test_run_dataset_execution_closure.py`
- `uv run pytest tests/sol_execbench/test_run_dataset_amd_score.py`
- `uv run --with ruff ruff check` on changed test files

## Non-Goals

Do not change production modules.

Do not change test behavior, expected values, helper semantics, or coverage
scope beyond moving tests to files that match their target modules.
