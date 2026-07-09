---
generated_by: gsd-map-codebase
generated_on: 2026-07-09
last_mapped_commit: cc007cd3af3e5100f7d86f155a40d5e51ffb57e5
focus: quality
---

# Testing

## Framework

Pytest is the test framework. The default configuration in `pyproject.toml` uses
`pytest-xdist` with `-n 8 --dist loadgroup`. A comment explicitly avoids
`-n auto` because each worker can load PyTorch+ROCm and consume significant RAM.

Common commands:

- `uv run pytest tests/`
- `uv run pytest tests/sol_execbench/test_e2e.py`
- `uv run pytest tests/sol_execbench/core/data/test_solution.py`
- `uv run --with ruff ruff check .`
- `uv run ty check`

## Marker Strategy

Markers are declared in both `pyproject.toml` and `tests/conftest.py`.
`tests/conftest.py` dynamically skips tests based on platform, architecture,
ROCm device availability, development headers, Python modules, and selected
marker expressions.

Important markers:

- `requires_linux`
- `requires_x86_64`
- `requires_rocm` and `requires_rocm_gpu`
- `requires_rocm_dev`
- `requires_triton_rocm`
- `requires_safetensors_torch`
- `requires_rdna4`
- `requires_cdna3`
- `requires_ck`
- `requires_rocwmma`
- `docker_dependency`
- `subprocess_uv`
- `native_extension`
- `native_extension_serial`
- `timing_serial`
- `requires_cutile` for legacy NVIDIA cuTile tests, skipped in this ROCm-only port.

`timing_serial`, `docker_dependency`, and `native_extension_serial` are skipped
by default unless explicitly selected with `-m`.

## Test Layout

The newer test layout mirrors package modules:

- `tests/sol_execbench/cli/` covers CLI commands, evaluation, reporting,
  runtime, timeout behavior, and sidecars.
- `tests/sol_execbench/core/bench/` covers correctness, timing, profiler,
  reward-hack checks, static kernel evidence, profile summaries, PID locking,
  and runtime helpers.
- `tests/sol_execbench/core/data/` covers definitions, workloads, dtypes,
  solutions, JSON utilities, and path access.
- `tests/sol_execbench/core/dataset/` covers migration, inventory, readiness,
  sharding, run closure, scoring reports, release readiness, and docs/claim
  guardrails.
- `tests/sol_execbench/core/platform/` covers diagnostics, dependency matrices,
  Docker matrices, toolchain routing, and ROCm migration residue.
- `tests/examples/` covers runnable examples and CLI path behavior.
- `tests/docker/dependencies/` covers container dependency expectations.

There are also legacy flat tests under `tests/sol_execbench/test_*.py` that
cover broad workflows and release/validation assertions.

## CI

`.github/workflows/code-quality.yml` runs on push and pull requests for Python
3.12 and 3.13. It installs with:

```bash
uv sync --locked --all-groups --python ${{ matrix.python-version }}
```

Then it runs Ruff, Ty, and CPU-safe pytest:

```bash
uv run pytest tests/sol_execbench \
  --ignore=tests/sol_execbench/driver/test_eval_driver.py \
  --ignore=tests/sol_execbench/test_e2e.py
uv run pytest tests/examples/test_examples.py -k consistency
```

Hardware ROCm behavior is not expected to run on GitHub-hosted CPU-only runners.

## Hardware-Sensitive Testing

ROCm availability detection in `tests/conftest.py` checks `/dev/kfd`,
`/dev/dri`, PyTorch HIP support, `torch.cuda.is_available()`, and the detected
AMD gfx architecture. RDNA4 detection uses `gfx12*`; CDNA3 detection uses
`gfx94*`.

Native extension tests also check `/opt/rocm/include/hip/hip_runtime_api.h`.
Composable Kernel and rocWMMA tests check `/opt/rocm/include/ck/ck.hpp` and
`/opt/rocm/include/rocwmma/rocwmma.hpp`.

## Example Coverage

Examples under `examples/` are covered by `tests/examples/` and selected core
dataset tests. They are intentionally excluded from Ruff formatting/linting,
because example source and JSON can be benchmark fixtures rather than package
style exemplars.

## Documentation And Claim Tests

Many tests assert documentation policy, public claim boundaries, release
readiness, provenance, and ROCm migration residue. Representative files include:

- `tests/sol_execbench/core/platform/test_rocm_migration_residue_audit.py`
- `tests/sol_execbench/core/dataset/test_public_prerelease_docs.py`
- `tests/sol_execbench/core/dataset/test_research_release_docs.py`
- `tests/sol_execbench/core/dataset/test_original_parity_docs.py`
- `tests/sol_execbench/core/dataset/test_rocm_library_readiness_docs.py`

Docs changes can break tests even when Python behavior is unchanged.

## Test Data And Fixtures

`tests/samples/` contains solution fixtures, including reward-hack examples and
legacy CUDA-labeled negative fixtures. `tests/sol_execbench_type_helpers.py`
contains typed helper constructors for schema tests.

Local downloaded benchmark assets belong under `data/` and should not be
committed unless deliberately curated as tiny fixtures.
