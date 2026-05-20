---
last_mapped: 2026-05-20
last_mapped_commit: unknown
focus: arch
---

# Structure

## Top-Level Layout

- `src/sol_execbench/` — package source.
- `tests/` — pytest suite, including unit, subprocess, Docker dependency, and
  sample/e2e tests.
- `examples/` — runnable sample problems for PyTorch, Triton, CUDA C++, CUTLASS,
  cuDNN, CuTe DSL, and cuTile.
- `docs/` — schema documentation for definitions, workloads, solutions, and
  traces.
- `scripts/` — dataset download, Docker launch, and dataset runner utilities.
- `docker/` — CUDA/cuDNN/CUTLASS development image and entrypoint.
- `data/` — local dataset mount point; contents are not source.

## Package Modules

- `src/sol_execbench/cli/main.py` contains the Click command implementation.
- `src/sol_execbench/driver/problem_packager.py` owns staging and command
  creation.
- `src/sol_execbench/driver/templates/` contains scripts copied into staging.
- `src/sol_execbench/core/data/` contains Pydantic schema models.
- `src/sol_execbench/core/bench/` contains runtime benchmark utilities.
- `src/sol_execbench/sol_score.py` contains the SOL score calculation.

## Test Layout

- `tests/sol_execbench/core/data/` validates schema models.
- `tests/sol_execbench/core/bench/` validates timing, correctness, IO, clock
  locking, and reward-hack helpers.
- `tests/sol_execbench/driver/` validates packaging, build script, and eval
  driver subprocess behavior.
- `tests/sol_execbench/samples/` contains fixtures for e2e tests.
- `tests/examples/test_examples.py` runs public examples.
- `tests/docker/dependencies/` checks container-level CUDA stack availability.

## Naming Patterns

Problem directories consistently contain `definition.json`, `workload.jsonl`,
and one or more `solution_*.json` files. Kernel source files are commonly named
`kernel.py`, `kernel.cu`, `main.cpp`, `main.cu`, or `reference.py`.

## Generated And Local Output

`data/`, `out/`, staging directories, caches, and build artifacts should remain
local. The CLI creates temporary directories named with the `sol_execbench_`
prefix unless `--keep-staging` is used.

## Planning Files

GSD planning artifacts are under `.planning/`. This map is stored in
`.planning/codebase/` and should be refreshed after significant structural
changes.
