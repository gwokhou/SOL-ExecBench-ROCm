---
last_mapped: 2026-05-20
last_mapped_commit: unknown
focus: quality
---

# Testing

## Framework

Tests use pytest with pytest-xdist. `pyproject.toml` sets
`addopts = "-n auto --dist loadgroup"`, so tests run in parallel by default with
load grouping.

## Main Commands

- `uv run pytest tests/` runs the full suite.
- `uv run pytest tests/sol_execbench/core/data/` runs schema tests.
- `uv run pytest tests/sol_execbench/driver/test_eval_driver.py` runs eval-driver
  subprocess tests.
- `uv run pytest tests -m timing_serial -n 0` runs serial GPU timing tests that
  are skipped by default.

## Markers And Fixtures

`tests/conftest.py` defines hardware-aware behavior:

- `requires_cutile` skips cuTile tests below `sm_100`.
- `requires_sm100` is available for Blackwell-only tests.
- `timing_serial` is skipped unless explicitly selected.
- `tmp_cache_dir` sets `SOLEXECBENCH_CACHE_PATH` to an isolated temp cache.

`pyproject.toml` also declares `cpp` for tests that compile C++/CUDA extensions.

## Coverage Areas

Data model tests under `tests/sol_execbench/core/data/` validate definitions,
solutions, workloads, and dtype handling. Benchmark tests under
`tests/sol_execbench/core/bench/` cover correctness, IO, timing, clock locking,
and reward-hack checks.

Driver tests under `tests/sol_execbench/driver/` exercise package staging,
`build_ext.py`, and `eval_driver.py`. E2E tests in `tests/sol_execbench/test_e2e.py`
load sample problems, run compile/execute phases, and assert passed traces.

## GPU And Docker Tests

`tests/docker/dependencies/` validates CUDA, Triton, CUTLASS, cuDNN, CuTe DSL,
and cuTile availability in the container. Many sample/e2e tests require a CUDA
GPU and compatible driver/toolchain.

## Test Data

Reusable sample problems are stored in `tests/sol_execbench/samples/` and public
examples in `examples/`. New features should add focused unit tests and, when
they affect evaluation behavior, a subprocess or sample-based integration test.
