---
type: research_note
status: archived
title: "Module-Aligned Layout Design"
source_format: superpowers
source_path: "docs/internal/superpowers/specs/2026-07-08-module-aligned-layout-design.md"
converted_at: "2026-07-09T00:00:00Z"
---

# Module-Aligned Layout Design

## Import Note

This research note was converted from a legacy Superpowers design spec. The original content is preserved below for traceability.

## Original Superpowers Document

# Module-Aligned Layout Design

## Goal

Reorganize SOL ExecBench so source files are grouped by module responsibility and
tests mirror the source tree where practical. Preserve existing public import paths
through compatibility facades during the move.

## Scope

This covers the tracked Python package under `src/sol_execbench/`, its package tests
under `tests/sol_execbench/`, and import references in scripts, examples, and docs
when they need to follow the new canonical layout.

The following locations keep their current top-level role:

- `tests/examples/` remains example workflow coverage.
- `tests/docker/` remains Docker environment coverage.
- `tests/sol_execbench/samples/` and `tests/sol_execbench/fixtures/` remain test data.
- `src/sol_execbench/data/` remains packaged data.

## Architecture

The repository already has strong module islands in `core/bench`, `core/data`,
`core/dataset`, and `core/scoring`. The reorganization should keep those islands and
focus on the remaining flat areas.

`src/sol_execbench/cli/` should be grouped by CLI concern:

- `cli/commands/` for command modules such as baseline, dataset, environment, and metadata.
- `cli/evaluation/` for evaluation execution, compilation, runtime, evaluator, problem I/O, and reporting.
- `cli/sidecars/` for sidecar creation and loading helpers.
- `cli/main.py` and `cli/__init__.py` remain stable entry points.

`src/sol_execbench/core/` should be grouped by domain:

- `core/reports/` for report-style domains such as claim upgrade, consistency, matrix diff,
  trust summary, evaluation stability, and report payloads.
- `core/platform/` for environment, toolchain, diagnostics, compatibility, dependency matrix,
  and Docker matrix modules.
- `core/evidence/` for shared artifact, checksum, baseline export, runtime evidence, and
  evidence reference helpers.

Existing import paths such as `sol_execbench.core.claim_upgrade` and
`sol_execbench.cli.evaluator` should continue to work as thin facade modules that re-export
from the new canonical modules. New internal imports should prefer the canonical module
paths once each move is complete.

## Test Layout

Tests should mirror source paths:

- `tests/sol_execbench/cli/commands/` mirrors `src/sol_execbench/cli/commands/`.
- `tests/sol_execbench/cli/evaluation/` mirrors `src/sol_execbench/cli/evaluation/`.
- `tests/sol_execbench/cli/sidecars/` mirrors `src/sol_execbench/cli/sidecars/`.
- `tests/sol_execbench/core/reports/` mirrors `src/sol_execbench/core/reports/`.
- `tests/sol_execbench/core/platform/` mirrors `src/sol_execbench/core/platform/`.
- `tests/sol_execbench/core/evidence/` mirrors `src/sol_execbench/core/evidence/`.

Already aligned tests under `tests/sol_execbench/core/bench`,
`tests/sol_execbench/core/data`, `tests/sol_execbench/core/scoring`, and
`tests/sol_execbench/driver` should remain in place unless a moved source file requires
an exact counterpart move.

## Migration Strategy

Move files in small module batches and run focused tests after each batch. For each moved
source module:

1. Move implementation code into the new canonical package.
2. Add or update `__init__.py` where needed.
3. Replace the old module with a facade that imports and re-exports the canonical symbols.
4. Move corresponding tests to the mirrored test path.
5. Update test imports to canonical paths where the test is about implementation details.
6. Leave public contract tests on facade paths when they intentionally verify compatibility.

## Verification

Run focused tests for each moved area, then run:

```bash
uv run pytest tests/sol_execbench/
uv run pytest tests/
uv run --with ruff ruff check .
```

GPU and ROCm-marked tests may skip or require hardware according to existing markers. The
reorganization is successful when Python imports, CLI entry points, scripts, and tests
continue to work through both canonical paths and preserved facades.
