<!-- generated-by: gsd-doc-writer -->
# Testing

Pytest is the project test framework. `pyproject.toml` configures
`pytest-xdist` with `-n auto --dist loadgroup`, so full-suite runs execute in
parallel by default.

## Setup

Install development dependencies:

```bash
uv sync --all-groups
```

The development group includes `pytest`, `pytest-xdist`, `ruff`, and `ty`.

## Core Commands

Run the full suite:

```bash
uv run pytest tests/
```

Run a focused package test:

```bash
uv run pytest tests/sol_execbench/test_e2e.py
```

Run schema tests:

```bash
uv run pytest tests/sol_execbench/core/data/
```

Run driver tests:

```bash
uv run pytest tests/sol_execbench/driver/
```

Run example consistency tests:

```bash
uv run pytest tests/examples/test_examples.py -k consistency
```

Run timing tests that are skipped by default:

```bash
uv run pytest tests -m timing_serial -n 0
```

Run Docker dependency checks inside the ROCm container:

```bash
uv run pytest tests/docker/dependencies/
```

## Markers

Markers are registered in `pyproject.toml` and `tests/conftest.py`.

| Marker | Meaning |
| --- | --- |
| `cpp` | Compiles HIP/C++ extensions and may be slow. |
| `timing_serial` | GPU timing tests skipped by default unless selected with `-m timing_serial`. |
| `requires_rocm` | Requires a ROCm GPU visible through PyTorch. |
| `requires_rocm_dev` | Requires ROCm native extension development headers under `/opt/rocm`. |
| `requires_ck` | Requires Composable Kernel headers under `/opt/rocm/include/ck/`. |
| `requires_rocwmma` | Requires rocWMMA headers under `/opt/rocm/include/rocwmma/`. |
| `requires_rdna4` | Requires an AMD RDNA 4 GPU such as `gfx1200`. |
| `requires_cdna3` | Requires an AMD CDNA 3 GPU such as `gfx942`. |
| `requires_cutile` | Legacy NVIDIA cuTile marker skipped in this ROCm-only port. |

On Linux, `requires_rocm` collection first checks for `/dev/kfd` and
`/dev/dri`. If those device nodes are not visible, the test is skipped with an
explicit diagnostic before importing or probing PyTorch ROCm.

## Hardware-Sensitive Tests

Run these only on a ROCm-capable Linux host or in a container with ROCm device
passthrough:

```bash
uv run pytest tests -m requires_rocm -n 0
uv run pytest tests -m requires_rdna4 -n 0
uv run pytest tests -m requires_cdna3 -n 0
uv run pytest tests -m requires_rocm_dev -n 0
```

For native library categories, use marker-filtered runs for the installed
headers:

```bash
uv run pytest tests -m requires_ck -n 0
uv run pytest tests -m requires_rocwmma -n 0
```

Docker dependency tests are hard readiness checks and should run only where ROCm
devices, user-space libraries, and container passthrough are expected:

```bash
./scripts/run_docker.sh --build
./scripts/run_docker.sh -- uv run pytest tests/docker/dependencies/
```

## ROCm Matrix Guardrails

The compatibility matrix and Docker target guardrails are designed to be
CPU-safe. They cover status classification, reason-code classification,
schema serialization, mixed-version blocking, claim flags, docs wording,
Docker Target selection, default behavior preservation, unknown Target rejection,
runtime evidence sidecars, dependency preflight behavior, dry-run wrapper
construction, and docs claim boundaries.

Focused matrix run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest \
  tests/sol_execbench/test_rocm_compatibility_matrix.py \
  tests/sol_execbench/test_matrix_claim_guardrails.py \
  tests/sol_execbench/test_docker_matrix_targets.py \
  tests/sol_execbench/test_docker_matrix_preflight.py \
  tests/sol_execbench/test_run_docker_matrix_script.py \
  tests/sol_execbench/test_dependency_matrix_policy.py \
  tests/sol_execbench/test_dependency_matrix_classification.py \
  tests/sol_execbench/test_dependency_matrix_cli.py \
  tests/sol_execbench/test_run_docker_dependency_preflight.py \
  tests/sol_execbench/test_runtime_evidence_reports.py \
  tests/sol_execbench/test_run_docker_runtime_evidence.py \
  tests/sol_execbench/test_rocm_matrix_docs.py -q
```

Wrapper syntax and docs lint checks:

```bash
bash -n scripts/run_docker.sh
UV_CACHE_DIR=/tmp/uv-cache uv run ruff check \
  docs/CLAIMS.md docs/TESTING.md tests/sol_execbench/test_rocm_matrix_docs.py
```

## Live ROCm Validation

Live ROCm validation is marker-gated. Use these checks only on a ROCm-capable
Linux host or container with ROCm device passthrough:

```bash
uv run pytest tests -m requires_rocm -n 0
uv run pytest tests -m requires_rdna4 -n 0
uv run pytest tests -m requires_cdna3 -n 0
```

For Matrix evidence, the current host ROCm 7.1.x environment may be recorded as
observed evidence through compatibility sidecars. ROCm 7.0.x or
ROCm 7.2.x native-host validation requires a matching host or separate machine;
default validation does not require host reinstall for ROCm 7.0.x or ROCm 7.2.x.
Docker rows for those Targets remain container ROCm user-space evidence on the
recorded host driver/devices unless direct native-host evidence is archived.

## Test Organization

| Area | Typical Coverage |
| --- | --- |
| `tests/sol_execbench/core/data/` | Pydantic schemas and JSON model behavior. |
| `tests/sol_execbench/core/bench/` | Correctness, timing, clock locking, reward-hack, and utility behavior. |
| `tests/sol_execbench/driver/` | Staging, build template, and evaluation driver behavior. |
| `tests/sol_execbench/test_*` | End-to-end, migration, public contract, scoring, environment, Docker, and docs guardrails. |
| `tests/examples/` | Example file consistency and runnable workflow coverage. |
| `tests/docker/dependencies/` | ROCm runtime, HIP, PyTorch ROCm, Triton ROCm, and library dependency checks. |

Use descriptive test names such as `test_rejects_invalid_solution_schema`, and
place new tests next to related coverage.

## Coverage

No coverage threshold is configured in `pyproject.toml`, and no coverage
configuration file is present.

| Type | Threshold |
| --- | --- |
| Lines | No threshold configured. |
| Branches | No threshold configured. |
| Functions | No threshold configured. |
| Statements | No threshold configured. |

## CI

The `Python Quality` workflow runs on push and pull request for Python 3.12 and
3.13:

```bash
uv sync --locked --all-groups
uv run ruff check .
uv run ty check
uv run pytest tests/sol_execbench \
  --ignore=tests/sol_execbench/driver/test_eval_driver.py \
  --ignore=tests/sol_execbench/test_e2e.py
uv run pytest tests/examples/test_examples.py -k consistency
```

The remote workflow excludes tests that need live ROCm GPU execution or Docker
runtime passthrough. Run those locally on suitable hardware before merging
hardware-sensitive changes.
