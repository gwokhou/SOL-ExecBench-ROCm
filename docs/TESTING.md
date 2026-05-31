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

## v1.19 Evidence Docs Guardrails

The centralized v1.19 guide is `docs/v1_19_evidence_guide.md`. Its focused
CPU-safe checks cover execution closure, paper denominator reports, Matrix
schema export, Matrix semantic diff, AMD bound sanity, and public wording
boundaries. v1.19 documentation has no full 235-problem paper validation, no
upstream SOLAR parity, no score authority, no leaderboard readiness, no CDNA
3/MI300X/CDNA4 validation, no native-host ROCm Matrix validation, and no
new-hardware validation.

Run the v1.19 documentation and sidecar-only contract checks:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest \
  tests/sol_execbench/test_research_release_docs.py \
  tests/sol_execbench/test_rocm_matrix_docs.py \
  tests/sol_execbench/test_public_contract_guardrails.py::test_v1_19_paper_denominator_fields_remain_sidecar_only \
  tests/sol_execbench/test_public_contract_guardrails.py::test_v1_19_amd_bound_sanity_fields_remain_sidecar_only -q

UV_CACHE_DIR=/tmp/uv-cache uv run ruff check \
  tests/sol_execbench/test_research_release_docs.py
```

This command set does not run GPU probes, ROCm live validation, Docker builds,
Docker containers, hardware-marker tests, dependency installs, or dependency
relocking.

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

### Compatibility Matrix Summary

| Target id | Local image tag | Requested ROCm user-space | Expected PyTorch ROCm stack | Current status | Recorded evidence |
| --- | --- | --- | --- | --- | --- |
| `rocm-7.0.2-ubuntu-24.04-container` | `sol-execbench:rocm-7.0.2-complete` | `7.0.2` | `torch==2.10.0+rocm7.0`, `torchvision==0.25.0+rocm7.0`, `triton-rocm==3.6.0` | `container_validated` | 2026-05-29 wrapper validation used `--record-container-validation` and wrote `.artifacts/e2e-260529/rocm-7.0.2-linear-wrapper-official.compatibility.json` with `status=container_validated`; `linear_backward` passed 3/3 workloads on RDNA 4 `gfx1200` / `AMD Radeon Graphics`; observed container ROCm user-space `7.0.2`, HIP `7.0.51831`, PyTorch `2.10.0+rocm7.0`, Triton `3.6.0`, and host devices `/dev/kfd` plus `/dev/dri`. Clock locking remains unavailable on ROCm 7.0.2, so performance evidence is unlocked with `CLOCKS_LOCKED=0`. |
| `rocm-7.1.1-ubuntu-24.04-container` | `sol-execbench:rocm-7.1.1-complete` | `7.1.1` | `torch==2.10.0+rocm7.1`, `torchvision==0.25.0+rocm7.1`, `triton-rocm==3.6.0` | `container_validated` | 2026-05-28 live Docker E2E on RDNA 4 `gfx1200` / `AMD Radeon Graphics`; observed HIP `7.1.25424`; `CLOCKS_LOCKED=1`; `linear_backward` passed 3/3 workloads. |
| `rocm-7.2.0-ubuntu-24.04-container` | `sol-execbench:rocm-7.2-complete` | `7.2.0` | `torch==2.11.0+rocm7.2`, `torchvision==0.26.0+rocm7.2`, `triton-rocm==3.6.0` | `container_validated` | 2026-05-29 wrapper validation used `--record-container-validation` and wrote `.artifacts/e2e-260529/rocm-7.2-linear-wrapper-official.compatibility.json` with `status=container_validated`; `linear_backward` passed 3/3 workloads on RDNA 4 `gfx1200` / `AMD Radeon Graphics`; observed container ROCm user-space `7.2.0`, HIP `7.2.26015`, PyTorch `2.11.0+rocm7.2`, Triton `3.6.0`, host devices `/dev/kfd` plus `/dev/dri`, and `CLOCKS_LOCKED=1`. |

The `container_validated` rows are container ROCm user-space evidence on the
recorded host driver/devices. It does not upgrade ROCm 7.1.x to native-host
validation. The `runtime_probe_passed` rows only show that the selected ROCm
container user-space can load the current host driver/runtime and see the GPU;
no current summary row remains at that limited status after the recorded Docker
smoke runs. Matrix rows are not clean project benchmark rows until their
target-specific Python dependency stack is installed, dependency preflight is
clean, and live Docker E2E is recorded. The Docker build now derives
target-specific PyTorch ROCm wheel arguments from `docker/rocm-targets.json`;
the recorded ROCm 7.0 and ROCm 7.2 validation rows used that path without mutating
the project lockfile.
The ROCm 7.0 row remains unlocked performance evidence because ROCm 7.0.2
reported ROCm SMI clock-command failure and ran with `CLOCKS_LOCKED=0`. The
ROCm 7.2 row completed with `CLOCKS_LOCKED=1`.

### E2E Execution Log

| Date | Target | Path | Status | Evidence |
| --- | --- | --- | --- | --- |
| 2026-05-28 | `rocm-7.1.1-ubuntu-24.04-container` | `./scripts/run_docker.sh --target ... --compatibility-entry ... --compatibility-matrix ... -- sol-execbench examples/pytorch/linear_backward ...` | Blocked by default matrix guardrail | Wrapper wrote `.artifacts/e2e-260528/rocm-7.1.1-linear-wrapper.compatibility.json` and `.artifacts/e2e-260528/rocm-7.1.1-linear-wrapper.matrix.json`, then stopped before benchmark because dependency preflight status `not_tested` has `benchmark_allowed=false`. The guardrail remains the default behavior. |
| 2026-05-28 | `rocm-7.1.1-ubuntu-24.04-container` | `./scripts/run_docker.sh --allow-untested-target-smoke --target ... --compatibility-entry ... --compatibility-matrix ... -- sol-execbench examples/pytorch/linear_backward ...` | Passed smoke, non-authoritative | Wrapper smoke ran through the explicit `not_tested` override and saved `.artifacts/e2e-260528/rocm-7.1.1-linear-wrapper-smoke.jsonl` with 3/3 `PASSED` workloads, HIP `7.1.25424`, PyTorch `2.10.0+rocm7.1`, Triton `3.6.0`, RDNA 4 `gfx1200` / `AMD Radeon Graphics`, and `CLOCKS_LOCKED=1`. The sidecar `.artifacts/e2e-260528/rocm-7.1.1-linear-wrapper-smoke.compatibility.json` remains `status=not_tested`, `benchmark_allowed=false`, and authority flags false. |
| 2026-05-28 | `rocm-7.1.1-ubuntu-24.04-container` | Direct `docker run ... sol-execbench examples/pytorch/linear_backward ...` | Passed | `sol-execbench:rocm-7.1.1-complete` reported HIP `7.1.25424`, PyTorch `2.10.0+rocm7.1`, Triton `3.6.0`, RDNA 4 `gfx1200` / `AMD Radeon Graphics`, and `CLOCKS_LOCKED=1`; `.artifacts/e2e-260528/rocm-7.1.1-linear-direct.jsonl` contains 3/3 `PASSED` workloads. |
| 2026-05-28 | `rocm-7.1.1-ubuntu-24.04-container` | Direct `docker run ... sol-execbench examples/hip_cpp/rmsnorm ... --static-evidence auto` | Passed | `sol-execbench:rocm-7.1.1-complete` compiled the HIP/C++ solution, reported `CLOCKS_LOCKED=1`, saved `.artifacts/e2e-260528/rocm-7.1.1-rmsnorm-hipcpp-direct.jsonl`, and collected `.artifacts/e2e-260528/rocm-7.1.1-rmsnorm-hipcpp-direct.jsonl.static-evidence.json`; 14/14 workloads `PASSED`. |
| 2026-05-28 | `rocm-7.1.1-ubuntu-24.04-container` | Direct `docker run ... uv run scripts/run_dataset.py tests/sol_execbench/samples/linear_backward ...` | Passed | Dataset runner path completed on the repository sample problem with `CLOCKS_LOCKED=1`; `.artifacts/e2e-260528/run-dataset-linear/summary.json` reports 1 problem OK and 3/3 workloads passed. The full `data/SOL-ExecBench/benchmark --limit 5` batch remains not run because the benchmark dataset is not present under `data/` on this machine. |
| 2026-05-28 | `rocm-7.0.2-ubuntu-24.04-container` | `docker build ... PYTORCH_TORCH_VERSION=2.10.0+rocm7.0 ...` then direct `docker run ... sol-execbench examples/pytorch/linear_backward ...` | Passed smoke, unlocked | Rebuild produced `sol-execbench:rocm-7.0.2-complete` and replaced the default ROCm 7.1 PyTorch wheels with `torch==2.10.0+rocm7.0` / `torchvision==0.25.0+rocm7.0`. `.artifacts/e2e-260528/rocm-7.0.2-linear-direct.jsonl` contains 3/3 `PASSED` workloads with HIP `7.0.51831` and Triton `3.6.0`. Clock locking was unavailable: ROCm 7.0.2 reported `Unable to set performance level to manual` and active MCLK stayed at level 0 after `--setmclk 1`, so `CLOCKS_LOCKED=0`. |
| 2026-05-28 | `rocm-7.2.0-ubuntu-24.04-container` | `docker build ... PYTORCH_TORCH_VERSION=2.11.0+rocm7.2 ...` then direct `docker run ... sol-execbench examples/pytorch/linear_backward ...` | Passed | Rebuild produced `sol-execbench:rocm-7.2-complete` and replaced the default ROCm 7.1 PyTorch wheels with `torch==2.11.0+rocm7.2` / `torchvision==0.26.0+rocm7.2`. `.artifacts/e2e-260528/rocm-7.2-linear-direct.jsonl` contains 3/3 `PASSED` workloads with HIP `7.2.26015`, Triton `3.6.0`, and `CLOCKS_LOCKED=1`. |
| 2026-05-29 | `rocm-7.2.0-ubuntu-24.04-container` | `./scripts/run_docker.sh --allow-mixed-version-dependencies --allow-untested-target-smoke --target ... --compatibility-entry ... --compatibility-matrix ... -- sol-execbench examples/pytorch/linear_backward ...` | Passed smoke, non-authoritative | Wrapper smoke ran through the explicit mixed-version diagnostic override and saved `.artifacts/e2e-260529/rocm-7.2-linear-wrapper-smoke.jsonl` with 3/3 `PASSED` workloads, HIP `7.2.26015`, PyTorch `2.11.0+rocm7.2`, Triton `3.6.0`, RDNA 4 `gfx1200` / `AMD Radeon Graphics`, and `CLOCKS_LOCKED=1`. The sidecar `.artifacts/e2e-260529/rocm-7.2-linear-wrapper-smoke.compatibility.json` remains `status=mixed_version` with authority flags false because the host project venv reports PyTorch `2.10.0+rocm7.1`. |
| 2026-05-29 | `rocm-7.2.0-ubuntu-24.04-container` | `./scripts/run_docker.sh --record-container-validation --target ... --compatibility-entry ... --compatibility-matrix ... -- sol-execbench examples/pytorch/linear_backward ...` | Passed, `container_validated` | Wrapper validation used target-container dependency evidence instead of the host venv and saved `.artifacts/e2e-260529/rocm-7.2-linear-wrapper-official.jsonl` with 3/3 `PASSED` workloads, HIP `7.2.26015`, PyTorch `2.11.0+rocm7.2`, Triton `3.6.0`, RDNA 4 `gfx1200` / `AMD Radeon Graphics`, and `CLOCKS_LOCKED=1`. The sidecar `.artifacts/e2e-260529/rocm-7.2-linear-wrapper-official.compatibility.json` records `status=container_validated`, container ROCm user-space `7.2.0`, toolchain ROCm `7.2.0`, and host devices `/dev/kfd` plus `/dev/dri`. |
| 2026-05-29 | `rocm-7.0.2-ubuntu-24.04-container` | `./scripts/run_docker.sh --record-container-validation --target ... --compatibility-entry ... --compatibility-matrix ... -- sol-execbench examples/pytorch/linear_backward ...` | Passed, `container_validated`, unlocked | Wrapper validation used target-container dependency evidence and saved `.artifacts/e2e-260529/rocm-7.0.2-linear-wrapper-official.jsonl` with 3/3 `PASSED` workloads, HIP `7.0.51831`, PyTorch `2.10.0+rocm7.0`, Triton `3.6.0`, RDNA 4 `gfx1200` / `AMD Radeon Graphics`, and `CLOCKS_LOCKED=0`. The sidecar `.artifacts/e2e-260529/rocm-7.0.2-linear-wrapper-official.compatibility.json` records `status=container_validated`, container ROCm user-space `7.0.2`, toolchain ROCm `7.0.2`, and host devices `/dev/kfd` plus `/dev/dri`; performance data remains unlocked because ROCm SMI clock lock fails under ROCm 7.0.2. |
| 2026-05-28 | `rocm-7.1.1-ubuntu-24.04-container` | Direct `docker run ... sol-execbench examples/hipblas/gemm ...` | Passed after include/link fix | Initial run failed at native extension compilation because the example omitted ROCm include/library paths and `c++` could not find `hip/hip_runtime_api.h`. After adding `-I/opt/rocm/include` and `-L/opt/rocm/lib`, hipBLAS GEMM compiled and ran with `CLOCKS_LOCKED=1`; `.artifacts/e2e-260528/rocm-7.1.1-hipblas-gemm.jsonl` contains 1/1 `PASSED` workload. |
| 2026-05-28 | `rocm-7.1.1-ubuntu-24.04-container` | Direct `docker run ... sol-execbench examples/miopen/softmax ...` | Passed | MIOpen softmax compiled and ran with `CLOCKS_LOCKED=1`; `.artifacts/e2e-260528/rocm-7.1.1-miopen-softmax.jsonl` contains 3/3 `PASSED` workloads. |
| 2026-05-28 | `rocm-7.1.1-ubuntu-24.04-container` | Direct `docker run ... sol-execbench examples/ck/gemm ...` | Passed | Composable Kernel GEMM compiled and ran with `CLOCKS_LOCKED=1`; `.artifacts/e2e-260528/rocm-7.1.1-ck-gemm.jsonl` contains 3/3 `PASSED` workloads. |
| 2026-05-28 | `rocm-7.1.1-ubuntu-24.04-container` | Direct `docker run ... sol-execbench examples/rocwmma/gemm ...` | Passed | rocWMMA GEMM compiled and ran with `CLOCKS_LOCKED=1`; `.artifacts/e2e-260528/rocm-7.1.1-rocwmma-gemm.jsonl` contains 3/3 `PASSED` workloads. |

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
