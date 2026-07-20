# Coupling and readability governance

## Required checks

```bash
uv run python scripts/check_coupling.py
uv run python scripts/check_readability.py
uv run python scripts/check_current_docs.py
uv run --with ruff ruff check .
uv run ty check
```

The coupling gate scans both `sol_execbench` and `solar`. It rejects:

- internal strongly connected import components;
- implementation imports from broad package facades;
- SOLAR importing benchmark code;
- benchmark code importing SOLAR outside `core.solar_bridge`;
- `core.platform` importing `core.scoring`;
- `core.bench` importing `core.reports`;
- generated candidate driver imports outside `driver.eval_runtime_api`;
- line-count or fan-out growth at named orchestration boundaries.

This repository performs clean internal refactors. Retired import paths are
removed rather than kept as thin compatibility facades. Update code, CLI,
tests and current documentation in the same change.

## Responsibility rules

- CLI modules coordinate user interaction; they do not own benchmark math.
- `ProblemPackager` stages assets; templates own process bootstrap only.
- The trusted reference service owns reference code, reference outputs and
  reference timing. Candidate modules cannot import reference runtime helpers.
- SOLAR owns formal analysis artifacts but never benchmark/candidate/scoring
  concepts. The bridge is the only cross-package adapter.
- Platform evidence is lower-level than scoring. Benchmark evidence producers
  are lower-level than report presentation.
- Formula, aggregation and official authority remain separate scoring modules.

## Readability debt

The standard baseline is non-increasing. SOLAR is additionally checked against
`scripts/solar_readability_debt.json`, an exact inventory of pre-existing long/wide
functions, `Any` modules and oversized modules. New debt or growth fails CI;
removing or shrinking an item passes without editing the inventory. Vendored
torchview code is excluded because it is not project-owned.

Passing import checks alone is not completion. Stages must be named, raw data
must stop at parsers, mutable orchestration state must be typed, subprocess I/O
must be bounded, and current docs/tests must match the implementation.
