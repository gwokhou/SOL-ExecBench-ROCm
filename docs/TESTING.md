<!-- generated-by: gsd-doc-writer -->
# Testing

Pytest is the project test framework. The test configuration in `pyproject.toml`
uses `pytest-xdist` with `-n auto --dist loadgroup`, so full-suite runs execute
tests in parallel by default.

## Test Framework And Setup

Install the development dependency group before running tests:

```bash
uv sync --all-groups
```

The `dev` dependency group in `pyproject.toml` includes:

| Package | Constraint |
| --- | --- |
| `pytest` | `>=9.0.2` |
| `pytest-xdist` | `>=3.5` |
| `ruff` | `>=0.4` |
| `ty` | `>=0.0.39` |

Some tests require ROCm hardware or compiler support. The configured pytest
markers in `pyproject.toml` and `tests/conftest.py` are:

| Marker | Meaning |
| --- | --- |
| `cpp` | Compiles HIP/C++ extensions and may be slow. |
| `timing_serial` | GPU timing tests skipped by default; run explicitly with `-m timing_serial -n 0`. |
| `requires_rocm` | Requires a ROCm GPU visible through PyTorch. |
| `requires_rocm_dev` | Requires ROCm native extension development headers under `/opt/rocm`. |
| `requires_ck` | Requires Composable Kernel headers under `/opt/rocm/include/ck/`. |
| `requires_rocwmma` | Requires rocWMMA headers under `/opt/rocm/include/rocwmma/`. |
| `requires_rdna4` | Requires an AMD RDNA 4 GPU such as `gfx1200`. |
| `requires_cdna3` | Requires an AMD CDNA 3 GPU such as `gfx942`. |
| `requires_cutile` | Legacy NVIDIA cuTile marker; skipped in this ROCm-only port. |

## Running Tests

Run the full suite:

```bash
uv run pytest tests/
```

Run one file:

```bash
uv run pytest tests/sol_execbench/test_e2e.py
```

Run schema-focused tests:

```bash
uv run pytest tests/sol_execbench/core/data/
```

Run driver-focused tests:

```bash
uv run pytest tests/sol_execbench/driver/
```

Run example consistency coverage:

```bash
uv run pytest tests/examples/test_examples.py -k consistency
```

Run timing tests that are skipped by default:

```bash
uv run pytest tests -m timing_serial -n 0
```

Run Docker dependency checks from inside the ROCm container:

```bash
uv run pytest tests/docker/dependencies/
```

## Writing New Tests

Place new package tests under `tests/sol_execbench/` next to related coverage.
Place example workflow tests under `tests/examples/`. Use descriptive names
such as `test_rejects_invalid_solution_schema`.

Existing test organization mirrors the source tree:

| Test Area | Typical Target |
| --- | --- |
| `tests/sol_execbench/core/data/` | Pydantic schema validation and JSON model behavior. |
| `tests/sol_execbench/core/bench/` | Correctness, timing, clock locking, IO, and reward-hack checks. |
| `tests/sol_execbench/driver/` | Staging, build template, and evaluation driver behavior. |
| `tests/sol_execbench/test_*` | End-to-end, migration, public-contract, and documentation guardrails. |
| `tests/examples/` | Example file consistency and runnable example workflows. |

Use markers for environment-sensitive coverage. Mark compiled extension tests
with `cpp`, ROCm GPU tests with `requires_rocm`, native header-dependent tests
with `requires_rocm_dev`, Composable Kernel tests with `requires_ck`, rocWMMA
tests with `requires_rocwmma`, architecture-specific tests with `requires_rdna4`
or `requires_cdna3`, and timing tests that should not run under xdist with
`timing_serial`.

## Coverage Requirements

No coverage threshold is configured in `pyproject.toml`, and no coverage config
file such as `.coveragerc`, `.nycrc`, or a pytest coverage threshold is present.

| Type | Threshold |
| --- | --- |
| Lines | No threshold configured. |
| Branches | No threshold configured. |
| Functions | No threshold configured. |
| Statements | No threshold configured. |

## CI Integration

No `.github/workflows/` directory is present in this checkout, so no GitHub
Actions test workflow is configured in the repository. Run the relevant local
pytest, `uv run ruff check .`, and `uv run ty check` commands before submitting
changes.

## GPU And Docker Checks

For hardware-sensitive work, validate the ROCm container and dependencies:

```bash
./scripts/run_docker.sh --build
./scripts/run_docker.sh -- uv run pytest tests/docker/dependencies/
```

For end-to-end benchmark behavior, run a small included example:

```bash
uv run sol-execbench examples/pytorch/gemma3_swiglu \
  --solution examples/pytorch/gemma3_swiglu/solution_python.json
```
