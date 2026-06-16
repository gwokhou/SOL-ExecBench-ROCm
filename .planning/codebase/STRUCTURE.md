---
generated_by: gsd-map-codebase
focus: arch
mapped_at: 2026-06-16
---

# Structure

## Top-Level Layout

| Path | Purpose |
| --- | --- |
| `src/sol_execbench/` | Python package source. |
| `tests/` | Pytest suite, examples tests, Docker dependency tests, and fixtures. |
| `examples/` | Runnable benchmark examples for PyTorch, Triton, HIP/C++, and ROCm libraries. |
| `scripts/` | Dataset, release, reporting, RDNA4, Docker, and setup helper scripts. |
| `docker/` | ROCm Dockerfile, entrypoint, and target metadata. |
| `docs/` | User docs, schema references, ROCm evidence docs, and release material. |
| `data/` | Local downloaded assets; excluded from normal source hygiene. |
| `.github/workflows/` | GitHub Actions quality workflow. |
| `.planning/` | GSD planning state, codebase maps, milestones, and project context. |

## Package Layout

| Path | Purpose |
| --- | --- |
| `src/sol_execbench/cli/main.py` | Main Click CLI, metadata commands, dataset migration commands, subprocess orchestration. |
| `src/sol_execbench/cli/baseline.py` | Public baseline trace comparison CLI. |
| `src/sol_execbench/core/data/` | Pydantic models for public benchmark inputs and outputs. |
| `src/sol_execbench/core/bench/` | Runtime helpers used by generated evaluation drivers. |
| `src/sol_execbench/core/dataset/` | Dataset layout, migration, readiness, closure, sharding, and runner helpers. |
| `src/sol_execbench/core/scoring/` | AMD SOL, AMD-native score, bound estimates, SOLAR derivation, and hardware models. |
| `src/sol_execbench/core/*.py` | Cross-cutting reporting, diagnostics, compatibility, matrix, trust, and utility modules. |
| `src/sol_execbench/driver/problem_packager.py` | Staging and command generation for compile/eval subprocesses. |
| `src/sol_execbench/driver/templates/` | Generated `build_ext.py` and `eval_driver.py` templates. |
| `src/sol_execbench/data/` | Packaged static AMD hardware model records. |

## Tests Layout

| Path | Purpose |
| --- | --- |
| `tests/conftest.py` | ROCm marker registration, skip logic, and fixtures. |
| `tests/sol_execbench/` | Main package tests for schemas, CLI behavior, dataset logic, scoring, reports, and guardrails. |
| `tests/sol_execbench/driver/` | Driver and packager tests. |
| `tests/sol_execbench/samples/` | Small benchmark fixtures used by tests. |
| `tests/examples/` | Example consistency and ROCm CLI path tests. |
| `tests/docker/dependencies/` | ROCm container dependency probes. |
| `tests/samples/` | Reward-hack sample solution manifests. |

## Scripts Layout

- Public helper scripts live at the top of `scripts/`, including
  `scripts/run_dataset.py`, `scripts/download_solexecbench.py`,
  `scripts/inspect_dataset.py`, and `scripts/run_docker.sh`.
- Internal RDNA4 evidence workflows live under `scripts/internal/rdna4/`.
- Internal release workflows live under `scripts/internal/release/`.
- Internal report generators live under `scripts/internal/reports/`.

## Examples Layout

The examples tree is grouped by solution category:

- `examples/pytorch/`
- `examples/triton/`
- `examples/hip_cpp/`
- `examples/hipblas/`
- `examples/miopen/`
- `examples/ck/`
- `examples/rocwmma/`
- Legacy NVIDIA-oriented examples are still present under `examples/cudnn/`,
  `examples/cutile/`, `examples/cutlass/`, and `examples/cute_dsl/` but are
  not accepted as ROCm solution categories.

Each runnable example generally contains `definition.json`, `workload.jsonl`,
one or more source/reference files, and a `solution_*.json` manifest.

## Naming Patterns

- Python modules and functions use `snake_case`.
- Pydantic models and enums use `PascalCase`.
- Tests use descriptive `test_*` names.
- Documentation uses uppercase canonical names for generated root docs under
  `docs/`, while many specialized docs use lowercase snake/kebab names.
- Dataset and evidence modules favor explicit domain names such as
  `profiler_timing_coverage.py`, `paper_denominator.py`, and
  `execution_closure.py`.

## Generated And Ignored Content

- `__pycache__/` files are present in the working tree listing but are not
  source files.
- `data/`, build artifacts, virtualenvs, and caches are excluded by Ruff.
- Benchmark outputs and downloaded datasets should not be committed.
