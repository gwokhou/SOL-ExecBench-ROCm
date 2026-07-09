---
type: research_note
status: archived
title: "Coupling Optimization Design"
source_format: superpowers
source_path: "docs/superpowers/specs/2026-07-07-coupling-optimization-design.md"
converted_at: "2026-07-09T00:00:00Z"
---

# Coupling Optimization Design

## Import Note

This research note was converted from a legacy Superpowers design spec. The original content is preserved below for traceability.

## Original Superpowers Document

# Coupling Optimization Design

## Goal

Reduce avoidable source-file coupling in `src/sol_execbench` while preserving
current public behavior and ROCm benchmark semantics.

## Scope

This work targets structural coupling found by static import analysis:

- Keep `core` independent from `cli`.
- Keep `core.data` as a foundational schema layer with no dependency on
  higher-level modules.
- Remove the cycle between SOLAR derivation model and coverage modules.
- Prevent accidental new cross-domain imports between `bench`, `dataset`, and
  `scoring`.
- Reduce overly broad package re-export surfaces where that can be done without
  breaking known callers.

This work does not redesign dataset execution, scoring formulas, profiler
behavior, or public schema semantics.

## Recommended Approach

Use boundary tests first, then apply small refactors:

1. Add an import-boundary test that parses internal imports with `ast` and
   asserts the intended layer rules.
2. Add explicit allowlists for existing justified cross-domain dependencies.
   The allowlist should be narrow and documented in the test.
3. Break the SOLAR derivation cycle by moving shared coverage/status model
   definitions or helpers into a one-way dependency location.
4. Trim `core.dataset` and `core.scoring` package exports only where usage
   proves the export is unnecessary or can be replaced by direct module imports.
5. Run focused tests after each refactor and the full relevant boundary suite at
   the end.

## Architecture

The target dependency direction is:

```text
cli -> core
core.{bench,dataset,scoring} -> core.data and small shared core utilities
core.data -> standard library / pydantic only
```

Some orchestration modules are allowed to bridge domains:

- `core.dataset.runner` may coordinate `bench` execution and scoring output.
- `core.dataset.amd_score_reports` may assemble scoring reports.
- selected `bench` artifact modules may use dataset checksum helpers until a
  dedicated shared checksum module is justified.

These exceptions must stay explicit in tests so future changes do not expand
them silently.

## Components

### Boundary Test

Create or extend a test under `tests/sol_execbench/` that:

- discovers `src/sol_execbench/**/*.py`;
- resolves absolute and relative `sol_execbench` imports to internal modules;
- checks forbidden layer edges;
- checks that no new two-node cycles appear except documented package import
  compatibility edges if still needed.

### SOLAR Coverage Refactor

The current cycle is:

```text
solar_derivation_models -> solar_derivation_coverage
solar_derivation_coverage -> solar_derivation_models
```

The refactor should make models independent of coverage implementation. Shared
coverage dataclasses should live with models or in a small dedicated module, and
coverage logic should depend on those dataclasses.

### Package Export Cleanup

`core.dataset.__init__` and `core.scoring.__init__` are aggregation points. Any
cleanup should be conservative:

- keep imports required by public or test callers;
- prefer direct imports in internal modules when practical;
- avoid breaking documented public APIs unless tests are updated to capture the
  intended replacement.

## Testing

Use TDD for every behavior or boundary change:

- Write the failing boundary test first.
- Run the focused test and confirm it fails for the current coupling issue.
- Apply the minimal refactor.
- Re-run the focused test until it passes.
- Run existing related tests:
  - `uv run pytest tests/sol_execbench/test_cli_module_boundaries.py`
  - `uv run pytest tests/sol_execbench/test_solar_derivation_contract.py`
  - `uv run pytest tests/sol_execbench/test_solar_derivation_evidence.py`
  - any tests directly affected by package export changes.

## Risks

- Package `__init__` cleanup may break callers that rely on re-exported names.
  Mitigation: search call sites before removing exports and keep compatibility
  where public usage is plausible.
- Boundary tests can become too rigid if they ban legitimate orchestration.
  Mitigation: use narrow allowlists with comments explaining each exception.
- Moving dataclasses can affect import paths if external users import them from
  the old module. Mitigation: preserve old import paths by re-exporting moved
  names when needed.

## Success Criteria

- The boundary test prevents the specific coupling regressions listed in scope.
- The SOLAR derivation model/coverage cycle is removed.
- No new broad cross-layer dependencies are introduced.
- Focused SOLAR, CLI boundary, and import-boundary tests pass.
