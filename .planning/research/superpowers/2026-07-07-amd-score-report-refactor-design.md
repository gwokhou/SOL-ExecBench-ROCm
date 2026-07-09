---
type: research_note
status: archived
title: "AMD Score Report Refactor Design"
source_format: superpowers
source_path: "docs/superpowers/specs/2026-07-07-amd-score-report-refactor-design.md"
converted_at: "2026-07-09T00:00:00Z"
---

# AMD Score Report Refactor Design

## Import Note

This research note was converted from a legacy Superpowers design spec. The original content is preserved below for traceability.

## Original Superpowers Document

# AMD Score Report Refactor Design

## Purpose

Reduce coupling in `src/sol_execbench/core/dataset/runner.py` by moving AMD score
report construction into a focused dataset module while preserving the existing
public import surface used by scripts and tests.

## Scope

Create `src/sol_execbench/core/dataset/amd_score_reports.py` for AMD score report
logic currently embedded in `runner.py`.

Move these functions and their direct dependencies:

- `build_amd_score_reports_for_problem`
- `write_amd_score_report`
- `_hardware_model_key_from_trace_payloads`
- `_read_json_object`
- `_minimal_amd_sol_bound_v2_from_payload`
- `_minimal_solar_aggregate_from_payload`

Keep these names available from `sol_execbench.core.dataset.runner` so existing
callers do not need to change.

## Compatibility

`scripts/run_dataset.py` and tests currently import score report helpers through
`runner.py`, and some tests monkeypatch `run_cli` through the script or runner
module. The refactor must not silently bypass those monkeypatches.

The implementation will keep a compatibility wrapper in `runner.py`. The wrapper
will delegate to the new module and pass the active `runner.run_cli` function into
the new implementation. This keeps monkeypatch behavior tied to the existing
module-level API.

## Architecture

`amd_score_reports.py` owns score report construction and file writing.
`runner.py` remains the dataset orchestration module and re-exports or wraps the
score report entry points.

The dependency direction should be:

- `runner.py` imports the new score report module.
- `amd_score_reports.py` imports scoring/data helpers it needs.
- `amd_score_reports.py` does not import `runner.py`.

This avoids a new cycle and keeps the extracted module independently testable.

## Testing

Use test-first implementation.

Add or adjust focused coverage to prove:

- `runner.build_amd_score_reports_for_problem` still exists.
- monkeypatching `runner.run_cli` still affects score report construction.
- existing AMD score report output paths and payload behavior remain unchanged.

Then run the relevant regression set:

- `uv run pytest tests/sol_execbench/test_run_dataset_amd_score.py`
- `uv run pytest tests/sol_execbench/test_dataset_runner.py`

## Non-Goals

Do not refactor score formulas, SOLAR derivation logic, CLI execution, trace
parsing, or `extend_derived_reports_for_problem` in this slice.

Do not change public schemas or report payload semantics.
