---
type: research_note
status: archived
title: "CLI Execution Refactor Design"
source_format: superpowers
source_path: "docs/internal/superpowers/specs/2026-07-07-cli-execution-refactor-design.md"
converted_at: "2026-07-09T00:00:00Z"
---

# CLI Execution Refactor Design

## Import Note

This research note was converted from a legacy Superpowers design spec. The original content is preserved below for traceability.

## Original Superpowers Document

# CLI Execution Refactor Design

## Purpose

Reduce coupling in `src/sol_execbench/core/dataset/runner.py` by moving dataset
CLI subprocess execution and CLI log handling into a focused module.

This repository does not currently need backward compatibility for internal
imports, so the refactor should update callers to import the new module directly
instead of keeping `runner.py` as a re-export surface.

## Scope

Create `src/sol_execbench/core/dataset/cli_execution.py`.

Move these names from `runner.py` into the new module:

- `CLI_LOG_LIMIT`
- `build_cli_command`
- `run_cli`
- `bounded_cli_stream`
- `bounded_file_stream`
- `_temporary_stream_path`
- `_parse_trace_jsonl`
- `save_cli_log`
- `save_cli_log_from_files`
- `save_cli_timeout_log`
- `save_cli_timeout_log_from_files`
- `cli_failure_notes`

Remove these runner-level compatibility aliases:

- `_bounded_cli_stream`
- `_save_cli_log`
- `_save_cli_timeout_log`
- `_cli_failure_notes`

## Callers

Update internal callers to import from `sol_execbench.core.dataset.cli_execution`
directly.

Expected caller updates:

- `scripts/run_dataset.py` imports CLI execution and logging helpers from the new
  module.
- `tests/sol_execbench/test_dataset_runner.py` tests the new module directly.
- Tests that currently exercise script-level aliases should either import the
  core helper directly or keep only aliases that are genuinely part of
  `scripts/run_dataset.py` behavior.

## Architecture

`cli_execution.py` owns subprocess invocation, bounded stream capture, temporary
stream files, JSONL trace parsing, failed CLI logs, timeout logs, and failure-note
parsing.

`runner.py` remains responsible for dataset solution building, timing evidence,
summary reporting, and derived report orchestration.

The dependency direction should be:

- `runner.py` may import `cli_execution.py` only if it still has direct local
  uses after extraction.
- `scripts/run_dataset.py` imports `cli_execution.py` directly.
- `cli_execution.py` does not import `runner.py`.

## Testing

Use test-first implementation.

Add or update focused coverage to prove:

- `build_cli_command` still builds the same command.
- `run_cli` still parses JSONL stdout, ignores non-JSON lines, passes the
  FlashInfer safetensors environment, writes logs on nonzero exit, filters benign
  ROCm stderr noise, and writes timeout logs.
- `cli_failure_notes` still recognizes exit-code failures and timeout logs.
- Script-level dataset execution tests continue to pass after import updates.

Run the relevant regression set:

- `uv run pytest tests/sol_execbench/test_dataset_runner.py`
- `uv run pytest tests/sol_execbench/test_run_dataset_execution_closure.py`
- `uv run pytest tests/sol_execbench/test_run_dataset_amd_score.py`
- `uv run --with ruff ruff check` on changed files

## Non-Goals

Do not change subprocess behavior, log file naming, log formatting, trace JSONL
parsing semantics, timeout handling, or ROCm stderr filtering.

Do not refactor AMD score report logic, timing evidence collection, dataset
execution closure logic, or solution-building helpers in this slice.
