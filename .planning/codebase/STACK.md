---
last_mapped_commit: f4de6692ee7468e150c112e0cbdcc8842dd0c709
mapped_at: 2026-06-06
---

# Technology Stack

## Runtime And Language

SOL ExecBench ROCm Port is a Python package with source under `src/sol_execbench/`.
The project requires Python `>=3.12,<3.14`, declared in `pyproject.toml`, and is
packaged with Hatchling via the `hatchling.build` backend.

The primary runtime is local Python subprocess execution for GPU benchmark
workloads. The CLI stages user solution files and generated drivers into a
temporary directory, then runs compile and evaluation commands through
`subprocess` from `src/sol_execbench/driver/problem_packager.py` and
`src/sol_execbench/cli/main.py`.

## Package Entry Points

- `sol-execbench` maps to `sol_execbench.cli:cli` in `pyproject.toml`.
- `sol-execbench-baseline` maps to `sol_execbench.cli.baseline:cli` in
  `pyproject.toml`.
- Main CLI implementation lives in `src/sol_execbench/cli/main.py`.
- Baseline artifact CLI implementation lives in `src/sol_execbench/cli/baseline.py`.
- Dataset-scale execution is provided as a script in `scripts/run_dataset.py`, with
  importable helpers in `src/sol_execbench/core/dataset/runner.py`.

The main CLI supports direct benchmark execution, `contract`, `doctor`,
`toolchain`, and `dataset` utility subcommands. Dataset utilities include local
SOL ExecBench migration and FlashInfer trace migration paths.

## Dependency Management

Dependencies are managed by uv with lockfile `uv.lock` and project metadata in
`pyproject.toml`.

Runtime dependencies:

- `torch==2.10.0` on non-Linux or non-x86_64 platforms.
- `torch==2.10.0+rocm7.1` on Linux x86_64.
- `torchvision==0.25.0` on non-Linux or non-x86_64 platforms.
- `torchvision==0.25.0+rocm7.1` on Linux x86_64.
- `triton-rocm==3.6.0` on Linux x86_64.
- `ninja>=1.13.0` for native extension builds.
- `pydantic>=2.12.5` for schema models.
- `safetensors>=0.7.0` for workload tensor assets.
- `click>=8.0` for CLI commands.
- `rich>=13.0` for human-readable CLI tables and progress.
- `datasets>=4.8.2` for Hugging Face dataset download.
- `torch-c-dlpack-ext>=0.1.5` and `apache-tvm-ffi>=0.1.9` for compatibility with
  low-level tensor/runtime integration paths.

Development dependencies in `pyproject.toml` are `pre-commit`, `pytest`,
`pytest-xdist`, `ruff`, and `ty`.

uv indexes in `pyproject.toml`:

- `pypi`: `https://pypi.org/simple`
- `pytorch-rocm71`: `https://download.pytorch.org/whl/rocm7.1`
- `pytorch-rocm-root`: `https://download.pytorch.org/whl/`

The ROCm-specific `torch`, `torchvision`, and `triton-rocm` packages are pinned to
these PyTorch wheel indexes through `tool.uv.sources`.

## Core Frameworks And Libraries

Schema and data models use Pydantic v2 models under
`src/sol_execbench/core/data/`. Important schema files include:

- `src/sol_execbench/core/data/definition.py` for problem definitions, axes,
  tensor specs, dtypes, and reference-code validation.
- `src/sol_execbench/core/data/workload.py` for random, scalar, custom, and
  safetensors workload inputs.
- `src/sol_execbench/core/data/solution.py` for solution source files, supported
  languages, target hardware, bindings, and compile options.
- `src/sol_execbench/core/data/trace.py` for evaluation trace output.
- `src/sol_execbench/core/data/contract.py` for published evaluator contract
  metadata.

GPU execution and timing use PyTorch ROCm. The code continues to use PyTorch's
historical `torch.cuda` API where it maps to HIP-backed ROCm devices and events,
not CUDA hardware. Timing helpers live in `src/sol_execbench/core/bench/timing.py`
and policy selection lives in `src/sol_execbench/core/bench/timing_policy.py`.

Native HIP/C++ extension builds use `torch.utils.cpp_extension` through the staged
template `src/sol_execbench/driver/templates/build_ext.py`. Evaluation uses the
staged Python driver `src/sol_execbench/driver/templates/eval_driver.py`.

## Supported Kernel Implementation Categories

The ROCm schema in `src/sol_execbench/core/data/solution.py` supports:

- `pytorch`
- `triton`
- `hip_cpp`
- `hipblas`
- `miopen`
- `ck`
- `rocwmma`

Legacy CUDA/NVIDIA schema values such as `cuda_cpp`, `cutlass`, `cudnn`,
`cublas`, `cute_dsl`, and `cutile` are rejected with migration guidance in
`src/sol_execbench/core/data/solution.py`.

Example problem directories under `examples/` demonstrate active and legacy
categories. ROCm-native examples include `examples/hip_cpp/`, `examples/hipblas/`,
`examples/miopen/`, `examples/ck/`, `examples/rocwmma/`, and `examples/triton/`.
Legacy NVIDIA-oriented examples remain under paths such as `examples/cudnn/`,
`examples/cutlass/`, `examples/cutile/`, and `examples/cute_dsl/` for migration
and compatibility coverage.

## ROCm Toolchain

ROCm host and container tooling is modeled explicitly:

- Runtime environment snapshots are implemented in
  `src/sol_execbench/core/environment.py`.
- Toolchain capability and routing metadata is implemented in
  `src/sol_execbench/core/toolchain.py`.
- ROCm profiler support is implemented in
  `src/sol_execbench/core/bench/rocm_profiler.py`.
- Static kernel evidence collection is implemented in
  `src/sol_execbench/core/bench/static_kernel_evidence.py`.
- GPU clock locking is implemented in `src/sol_execbench/core/bench/clock_lock.py`
  and configured in `src/sol_execbench/core/bench/config/device_config.py`.

External ROCm commands probed or invoked include `amd-smi`, `rocm-smi`,
`rocminfo`, `rocm_agent_enumerator`, `rocprofv3`, and `rocprofv3-avail`.

HIP compilation target detection in `src/sol_execbench/driver/problem_packager.py`
uses `rocm_agent_enumerator -name` first, then `rocminfo`, to derive a local
`gfx*` target. If the solution targets `LOCAL` and no explicit offload
architecture flag is present, the packager injects HIP flags such as
`--offload-arch=gfx1200`.

Packaged AMD hardware model data currently lives in
`src/sol_execbench/data/amd_hardware_models/gfx1200.json` and is loaded by
`src/sol_execbench/core/scoring/amd_hardware_models.py`.

## Docker And GPU Environment

Docker support lives in `docker/`:

- `docker/Dockerfile` builds on `rocm/dev-ubuntu-24.04` and defaults to tag
  `7.1.1-complete`.
- `docker/entrypoint.sh` detects HIP-backed PyTorch, attempts clock locking, and
  unlocks clocks on process exit.
- `docker/rocm-targets.json` records supported container targets for ROCm 7.0.2,
  7.1.1, and 7.2.0.
- `scripts/run_docker.sh` is the user-facing Docker runner.

The Dockerfile installs ROCm runtime tooling, Python, build tools, uv, PyTorch
ROCm wheels, torchvision ROCm wheels, and `triton-rocm`. It sets:

- `ROCM_PATH=/opt/rocm`
- `HIP_PATH=/opt/rocm`
- `HIP_PLATFORM=amd`
- `PATH=/opt/rocm/bin:/opt/rocm/llvm/bin:${PATH}`
- `LD_LIBRARY_PATH=/opt/rocm/lib`
- `UV_PROJECT_ENVIRONMENT=/venv`
- `PYTHONPATH=/sol-execbench/src`

The README documents ROCm device requirements through `/dev/kfd` and `/dev/dri`.

## Data And Dataset Processing

Dataset code is under `src/sol_execbench/core/dataset/`. It covers categories,
layout inspection, manifests, migration, readiness classification, sharding,
execution closure, run state, and scoring/report sidecars.

The public SOL ExecBench dataset downloader is `scripts/download_solexecbench.py`.
It uses `datasets.load_dataset` with repo ID `nvidia/SOL-ExecBench`, writes local
problem directories under `data/SOL-ExecBench/benchmark/`, and can write a
deterministic manifest through `src/sol_execbench/core/dataset/manifest.py`.

Dataset-scale runs in `scripts/run_dataset.py` can run either one problem
directory or a dataset root with categories `L1`, `L2`, `FlashInfer-Bench`, and
`Quant`. They generate derived summaries, AMD score reports, timing evidence,
and execution closure metadata.

Safetensors workload support lives in `src/sol_execbench/core/bench/io.py`.
The code uses `FLASHINFER_TRACE_DIR` to find FlashInfer trace assets when
relative safetensors paths are not located in the repo root.

## Scoring And Reporting

Scoring logic is under `src/sol_execbench/core/scoring/`, including AMD-native
score reports, AMD SOL bounds, AMD SOL v2 artifacts, bound sanity checks,
hardware models, and solar derivation evidence.

Reporting and evidence utilities include:

- `src/sol_execbench/core/reporting.py`
- `src/sol_execbench/core/runtime_evidence.py`
- `src/sol_execbench/core/evaluation_stability.py`
- `src/sol_execbench/core/consistency.py`
- `src/sol_execbench/core/trust_summary.py`
- `src/sol_execbench/core/claim_upgrade.py`
- `src/sol_execbench/core/matrix_diff.py`
- `src/sol_execbench/core/dependency_matrix.py`
- `src/sol_execbench/core/docker_matrix.py`

Script entry points for report generation live in `scripts/`, including
`scripts/report_amd_bound_sanity.py`, `scripts/report_claim_upgrade.py`,
`scripts/report_consistency.py`, `scripts/report_evaluation_stability.py`,
`scripts/report_paper_denominator.py`, `scripts/report_parity_gaps.py`,
`scripts/report_trust_summary.py`, `scripts/export_matrix_schema.py`,
`scripts/diff_matrix_reports.py`, and
`scripts/build_prerelease_artifact_bundle.py`.

## Configuration Surfaces

Primary project configuration is in `pyproject.toml`.

Runtime benchmark config is represented by
`src/sol_execbench/core/bench/config/benchmark_config.py` and can be passed to
the CLI with `--config`. Problem directories may contain `config.json`.

Environment variables used by runtime paths include:

- `SOLEXECBENCH_ENV_SNAPSHOT` and `SOLEXECBENCH_ENV_SNAPSHOT_PATH` in
  `src/sol_execbench/cli/main.py`.
- `FLASHINFER_TRACE_DIR` in `src/sol_execbench/core/bench/io.py`,
  `src/sol_execbench/driver/problem_packager.py`, and `docker/entrypoint.sh`.
- `PYTORCH_ALLOC_CONF=expandable_segments:True` set for evaluation subprocesses in
  `src/sol_execbench/cli/main.py`.
- `SOL_EXECBENCH_CLOCKS_LOCKED` set by `docker/entrypoint.sh`.
- `SOL_EXECBENCH_RUNTIME_GFX_ARCHITECTURE` and `PYTORCH_ROCM_ARCH` used by
  `scripts/run_dataset.py`.
- Device visibility variables captured by `src/sol_execbench/core/environment.py`:
  `HIP_VISIBLE_DEVICES`, `ROCR_VISIBLE_DEVICES`, and `HSA_OVERRIDE_GFX_VERSION`.

## Testing And Quality Tooling

Pytest configuration in `pyproject.toml` runs with `pytest-xdist` using
`-n auto --dist loadgroup`. Markers include `cpp`, `requires_rocm`,
`requires_rdna4`, `requires_cdna3`, and legacy `requires_cutile`.

Linting and formatting use Ruff, configured in `pyproject.toml`. Type checking
uses `ty` with `src` and `tests` included.

Pre-commit hooks are configured in `.pre-commit-config.yaml`:

- Ruff check with fixes on pre-commit.
- Ruff format on pre-commit.
- DCO sign-off check on commit messages.
- `ty check` on pre-push.

GitHub Actions CI is defined in `.github/workflows/code-quality.yml`. It runs on
Python 3.12 and 3.13, installs with `uv sync --locked --all-groups`, then runs
Ruff, Ty, CPU-safe package tests, and example consistency tests.
