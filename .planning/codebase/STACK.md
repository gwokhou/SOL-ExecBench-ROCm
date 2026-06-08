---
last_mapped_commit: dd5731c42d9f3d417acf552b3a02004cd5039df2
last_mapped_date: 2026-06-08
focus: tech
---

# Technology Stack

## Runtime And Packaging

- Python package: `sol-execbench`, source under `src/sol_execbench/`.
- Python version: `>=3.12,<3.14` in `pyproject.toml`.
- Build backend: Hatchling via `[build-system]` in `pyproject.toml`.
- Environment and lock workflow: uv with `uv.lock`, project indexes, and dependency groups in `pyproject.toml`.
- Console entry points:
  - `sol-execbench` -> `sol_execbench.cli:cli`
  - `sol-execbench-baseline` -> `sol_execbench.cli.baseline:cli`

## Core Python Dependencies

- PyTorch ROCm is the main tensor/runtime backend:
  - Linux x86_64 pins `torch==2.10.0+rocm7.1` and `torchvision==0.25.0+rocm7.1`.
  - Non-Linux/non-x86_64 pins plain `torch==2.10.0` and `torchvision==0.25.0`.
  - uv sources point ROCm wheels to `https://download.pytorch.org/whl/rocm7.1`.
- `triton-rocm==3.6.0` supports Triton solution examples and ROCm JIT kernels.
- `pydantic>=2.12.5` defines strict benchmark schemas and evidence payloads.
- `click>=8.0` and `rich>=13.0` implement the CLI.
- `safetensors>=0.7.0` supports large benchmark input blobs.
- `datasets>=4.8.2` supports dataset-oriented workflows.
- `ninja>=1.13.0` supports native extension builds through PyTorch.
- `torch-c-dlpack-ext>=0.1.5` and `apache-tvm-ffi>=0.1.9` are packaged runtime dependencies for extended benchmark compatibility.

## Development Tooling

- Test runner: pytest, with parallel execution via `pytest-xdist`, configured in `pyproject.toml`.
- Lint/format: Ruff, with generated data and `examples/` excluded in `pyproject.toml`.
- Type checking: ty configured for `src` and `tests`.
- Pre-commit is included in the dev dependency group.
- Test marker logic for ROCm hardware and headers lives in `tests/conftest.py`.

## Application Shape

- CLI orchestration lives in `src/sol_execbench/cli/main.py`.
- Baseline CLI logic lives in `src/sol_execbench/cli/baseline.py`.
- Pydantic models and public JSON contracts live in `src/sol_execbench/core/data/`.
- Benchmark execution helpers live in `src/sol_execbench/core/bench/`.
- Problem staging and native build orchestration live in `src/sol_execbench/driver/problem_packager.py`.
- Staged evaluator template lives in `src/sol_execbench/driver/templates/eval_driver.py`.
- Native extension build template lives in `src/sol_execbench/driver/templates/build_ext.py`.
- Dataset migration, readiness, run-state, and parity helpers live in `src/sol_execbench/core/dataset/`.
- Scoring and AMD-bound analysis live in `src/sol_execbench/core/scoring/`.
- Reporting, stability, trust, compatibility, and guardrail utilities live in `src/sol_execbench/core/`.

## Benchmark Data Contracts

- Problem inputs are file-based:
  - `definition.json`
  - `workload.jsonl`
  - optional `config.json`
  - `solution.json`
- Canonical data models include:
  - `src/sol_execbench/core/data/definition.py`
  - `src/sol_execbench/core/data/workload.py`
  - `src/sol_execbench/core/data/solution.py`
  - `src/sol_execbench/core/data/trace.py`
  - `src/sol_execbench/core/data/contract.py`
- Evaluation output is canonical JSONL `Trace` data emitted by `src/sol_execbench/driver/templates/eval_driver.py`.
- Diagnostic-only sidecars are separate from canonical traces; examples include no-trace diagnostics, environment snapshots, static kernel evidence, and `rocprofv3` artifacts.

## GPU Runtime Stack

- ROCm is the only active GPU platform.
- Supported target hardware values are defined in `src/sol_execbench/core/data/solution.py`:
  - `gfx1200` for RDNA 4.
  - `gfx940`, `gfx941`, and `gfx942` for CDNA 3.
  - `LOCAL` for detected local AMD GPU target.
- Runtime tensor execution uses PyTorch's HIP backend through the `torch.cuda` API.
- Timing uses HIP-backed PyTorch events by default in `src/sol_execbench/core/bench/timing.py` and policy logic in `src/sol_execbench/core/bench/timing_policy.py`.
- Optional profiler-backed timing and profile collection uses `rocprofv3` in `src/sol_execbench/core/bench/rocm_profiler.py`.
- GPU clock locking uses `rocm-smi` through passwordless `sudo -n` in `src/sol_execbench/core/bench/clock_lock.py`.

## Supported Solution Technologies

- Language and library categories are defined in `src/sol_execbench/core/data/solution.py`.
- Active solution categories:
  - `pytorch`
  - `triton`
  - `hip_cpp`
  - `hipblas`
  - `miopen`
  - `ck`
  - `rocwmma`
- Legacy NVIDIA categories such as `cuda_cpp`, `cublas`, `cudnn`, `cutlass`, `cute_dsl`, and `cutile` are rejected or represented only by compatibility examples/docs.
- Native ROCm solution examples are under:
  - `examples/hip_cpp/`
  - `examples/hipblas/`
  - `examples/miopen/`
  - `examples/ck/`
  - `examples/rocwmma/`
  - `examples/triton/`
  - `examples/pytorch/`

## Native Build Path

- HIP/C++ solutions are staged by `src/sol_execbench/driver/problem_packager.py`.
- Compilation runs `src/sol_execbench/driver/templates/build_ext.py` inside the staging directory.
- Native builds use `torch.utils.cpp_extension.load`.
- Device compiler flags are passed with PyTorch's `extra_cuda_cflags` keyword, which is also used for ROCm.
- `PYTORCH_ROCM_ARCH` is set from declared `gfx*` targets when not already present.
- Offload architecture flags are auto-injected for native ROCm builds when no explicit `--offload-arch`, `-offload-arch`, or `--amdgpu-target` flag is present.
- Compile options are restricted to prevent path injection, response files, runtime linker control, and host escape flags.

## Docker And Container Stack

- Runtime container image is built from ROCm dev images in `docker/Dockerfile`.
- Default Docker target is declared in `docker/rocm-targets.json`:
  - `rocm-7.1.1-ubuntu-24.04-container`
  - base image `rocm/dev-ubuntu-24.04:7.1.1-complete`
  - PyTorch ROCm target `rocm7.1`
- Additional declared Docker targets cover ROCm 7.0.2 and 7.2.0 in `docker/rocm-targets.json`.
- `docker/Dockerfile` installs system packages, uv, Python dependencies, PyTorch ROCm wheels, and `triton-rocm`.
- `docker/entrypoint.sh` probes PyTorch HIP, locks clocks when available, and unlocks on exit.
- `scripts/run_docker.sh` selects Docker targets, performs runtime/dependency preflight checks, mounts ROCm devices, and runs commands inside the container.
- Docker matrix logic lives in `src/sol_execbench/core/docker_matrix.py`.
- Dependency policy and preflight classification live in `src/sol_execbench/core/dependency_matrix.py`.

## External Tool Probes

- Environment snapshots in `src/sol_execbench/core/environment.py` probe:
  - `amd-smi`
  - `rocminfo`
  - `rocm_agent_enumerator`
  - PyTorch ROCm availability
- Toolchain routing in `src/sol_execbench/core/toolchain.py` models and probes:
  - `rocprofv3`
  - `rocprofv3-avail`
  - migrated/deprecated ROCm profiler tooling
  - static artifact tool candidates
- Native staging detects local `gfx*` targets with `rocm_agent_enumerator -name`, falling back to `rocminfo`.
- Dependency checks probe `hipcc --version` and ROCm version files under `/opt/rocm/.info/`.

## Scripts And Operational Tooling

- Dataset and batch execution: `scripts/run_dataset.py`.
- Dataset download/migration helpers: `scripts/download_data.sh`, `scripts/download_solexecbench.py`, and CLI migration commands in `src/sol_execbench/cli/main.py`.
- Docker launcher: `scripts/run_docker.sh`.
- Sudoers helper for clock control: `scripts/setup_rocm_smi_sudoers.py`.
- Evidence/report scripts:
  - `scripts/report_amd_bound_sanity.py`
  - `scripts/report_consistency.py`
  - `scripts/report_evaluation_stability.py`
  - `scripts/report_paper_denominator.py`
  - `scripts/report_trust_summary.py`
  - `scripts/report_claim_upgrade.py`
  - `scripts/diff_matrix_reports.py`
  - `scripts/export_matrix_schema.py`
- Release and prerelease tooling:
  - `scripts/check_prerelease_readiness.py`
  - `scripts/build_prerelease_artifact_bundle.py`
  - `scripts/release_candidate_validation.py`

## Documentation Stack

- Public docs are Markdown under `docs/`.
- ROCm-facing technical docs include:
  - `docs/rocm.md`
  - `docs/rocm_timing.md`
  - `docs/rocm_toolchain_routing.md`
  - `docs/static_kernel_evidence.md`
  - `docs/rocm_libraries.md`
  - `docs/TESTING.md`
  - `docs/CONFIGURATION.md`
  - `docs/RESEARCHER-GUIDE.md`
- Internal validation notes are under `docs/internal/`.
- Example evidence artifacts are under `docs/examples/`.
