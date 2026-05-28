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

On Linux, `requires_rocm` collection first checks for `/dev/kfd` and `/dev/dri`.
If those device nodes are hidden by the current execution environment, such as a
Codex or container sandbox without ROCm device passthrough, the test is skipped
with an explicit device-node diagnostic before PyTorch probes the GPU runtime.

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

These Docker dependency checks intentionally remain hard readiness checks. They
should run only where ROCm devices are passed through, and they may fail instead
of skipping when `/dev/kfd` or `/dev/dri` is unavailable.

## ROCm Matrix Guardrails

The ROCm Compatibility Matrix guardrails are CPU-safe by default. They cover
status classification, reason-code classification, schema serialization,
mixed-version blocking, claim flags, docs wording, Docker Target selection,
default behavior preservation, unknown Target rejection, and wrapper command
construction without requiring live ROCm hardware.

Run the focused matrix guardrail suite:

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

Run wrapper syntax and docs lint checks:

```bash
bash -n scripts/run_docker.sh
UV_CACHE_DIR=/tmp/uv-cache uv run ruff check \
  docs/CLAIMS.md docs/TESTING.md tests/sol_execbench/test_rocm_matrix_docs.py
```

The matrix guardrail suite intentionally uses injected observations, dry-run
script execution, and sidecar assertions. It must not require Docker daemon
access, `/dev/kfd`, `/dev/dri`, PyTorch ROCm imports, or a live GPU.

Live ROCm validation is marker-gated. Use these checks only on a ROCm-capable
Linux host or container with ROCm device passthrough:

```bash
uv run pytest tests -m requires_rocm -n 0
uv run pytest tests -m requires_rdna4 -n 0
uv run pytest tests -m requires_cdna3 -n 0
```

For v1.18 Matrix evidence, the current host ROCm 7.1.x environment may be
recorded as observed evidence through compatibility sidecars. ROCm 7.0.x or
ROCm 7.2.x native-host validation requires a matching host or separate machine;
default validation does not require host reinstall for ROCm 7.0.x or ROCm 7.2.x.
Docker rows for those Targets remain container ROCm user-space evidence on the
recorded host driver/devices unless direct native-host evidence is archived.

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

The repository includes a GitHub Actions quality workflow at
`.github/workflows/code-quality.yml`. It runs on pushes and pull requests across
Python 3.12 and 3.13, installs with `uv sync --locked --all-groups`, then runs
Ruff linting, Ty type checks, and CPU-safe pytest coverage:

```bash
uv run ruff check .
uv run ty check
uv run pytest tests/sol_execbench \
  --ignore=tests/sol_execbench/driver/test_eval_driver.py \
  --ignore=tests/sol_execbench/test_e2e.py
uv run pytest tests/examples/test_examples.py -k consistency
```

The remote workflow intentionally excludes `tests/docker/dependencies/` because
those tests validate real ROCm runtime, container, and device passthrough
readiness. It also excludes eval-driver and end-to-end execution tests that
expect a visible ROCm device for timing. Run those checks in a ROCm-capable
Docker environment before merging hardware-sensitive changes.

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
