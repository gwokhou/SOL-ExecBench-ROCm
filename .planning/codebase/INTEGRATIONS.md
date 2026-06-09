---
last_mapped_commit: dd5731c42d9f3d417acf552b3a02004cd5039df2
last_mapped_date: 2026-06-08
focus: tech
---

# External Integrations

## PyTorch ROCm

- Role: primary tensor runtime, HIP device interface, timing event backend, and native extension build provider.
- Dependency declaration: `pyproject.toml`.
- ROCm wheel indexes:
  - `https://download.pytorch.org/whl/rocm7.1`
  - `https://download.pytorch.org/whl/`
- Runtime usage:
  - `src/sol_execbench/driver/templates/eval_driver.py`
  - `src/sol_execbench/core/bench/timing.py`
  - `src/sol_execbench/core/bench/eval_runtime.py`
  - `src/sol_execbench/core/bench/io.py`
  - `src/sol_execbench/core/environment.py`
- Native extension usage:
  - `src/sol_execbench/driver/templates/build_ext.py`
  - `src/sol_execbench/driver/problem_packager.py`
- Detection and policy:
  - `src/sol_execbench/core/dependency_matrix.py`
  - `docker/rocm-targets.json`
  - `scripts/run_docker.sh`

## Triton ROCm

- Role: supported solution category for generated GPU kernels.
- Dependency declaration: `triton-rocm==3.6.0` in `pyproject.toml`.
- Docker install: `docker/Dockerfile`.
- Examples:
  - `examples/triton/rmsnorm/`
  - `examples/triton/olmo3_post_norm/`
  - `examples/triton/nemotron_rms_norm/`
- Timing policy explicitly distinguishes Triton kernels in `src/sol_execbench/core/bench/timing_policy.py` and `docs/rocm_timing.md`.

## ROCm User-Space Tools

- `rocminfo`
  - Used for environment snapshots in `src/sol_execbench/core/environment.py`.
  - Used as local `gfx*` fallback detector in `src/sol_execbench/driver/problem_packager.py`.
  - Checked during container build in `docker/Dockerfile`.
- `rocm_agent_enumerator`
  - Used for environment snapshots in `src/sol_execbench/core/environment.py`.
  - Preferred local `gfx*` detector in `src/sol_execbench/driver/problem_packager.py`.
- `hipcc`
  - Checked by dependency preflight in `src/sol_execbench/core/dependency_matrix.py`.
  - Checked during container build in `docker/Dockerfile`.
- ROCm version files
  - `/opt/rocm/.info/version`
  - `/opt/rocm/.info/version-dev`
  - Read by `src/sol_execbench/core/dependency_matrix.py`.

## AMD SMI And ROCm SMI

- `amd-smi`
  - Probed for optional environment evidence in `src/sol_execbench/core/environment.py`.
  - Checked as an available system tool in `docker/Dockerfile`.
  - Used for GPU clock lock, passwordless sudo probing, and unlock in
    `src/sol_execbench/core/bench/clock_lock.py`.
  - Sudoers setup is handled in `docker/Dockerfile` and
    `scripts/setup_rocm_clock_sudoers.py`.
- `rocm-smi`
  - Used for read-only clock/performance-level verification in
    `src/sol_execbench/core/bench/clock_lock.py`.
  - Container entrypoint calls clock lock/unlock via `docker/entrypoint.sh`.
- Environment knobs:
  - `SOL_EXECBENCH_CLOCKS_LOCKED`

## ROCprofiler SDK

- `rocprofv3`
  - Optional diagnostic profiling and timing evidence collection in `src/sol_execbench/core/bench/rocm_profiler.py`.
  - CLI profile mode is wired in `src/sol_execbench/cli/main.py`.
  - Toolchain routing registry entry is in `src/sol_execbench/core/toolchain.py`.
  - Documented in `docs/rocm_timing.md` and `docs/rocm_toolchain_routing.md`.
- `rocprofv3-avail`
  - Modeled as a counter/config discovery companion in `src/sol_execbench/core/toolchain.py`.
- `rocprofiler-systems` and `rocm-systems`
  - Modeled as migrated/deprecated routing context in `src/sol_execbench/core/toolchain.py`.
- Profiler output is diagnostic-only and does not replace canonical trace JSONL.

## Docker

- Role: reproducible ROCm benchmark environment with GPU device passthrough.
- Dockerfile: `docker/Dockerfile`.
- Entrypoint: `docker/entrypoint.sh`.
- Launcher: `scripts/run_docker.sh`.
- Target manifest: `docker/rocm-targets.json`.
- Default base image: `rocm/dev-ubuntu-24.04:7.1.1-complete`.
- Declared targets:
  - `rocm-7.0.2-ubuntu-24.04-container`
  - `rocm-7.1.1-ubuntu-24.04-container`
  - `rocm-7.2.0-ubuntu-24.04-container`
- Device integration:
  - `/dev/kfd`
  - `/dev/dri`
- Docker preflight can use environment overrides such as:
  - `SOL_EXECBENCH_DOCKER_CONTEXT`
  - `SOL_EXECBENCH_DOCKER_HOST`
  - `SOL_EXECBENCH_DEV_KFD_PRESENT`
  - `SOL_EXECBENCH_DEV_DRI_PRESENT`
  - `SOL_EXECBENCH_GPU_ACCESSIBLE`

## Native ROCm Libraries

- hipBLAS
  - Supported language category `hipblas` in `src/sol_execbench/core/data/solution.py`.
  - Example: `examples/hipblas/gemm/`.
  - Typical compile option includes `-lhipblas` in solution JSON.
- MIOpen
  - Supported language category `miopen` in `src/sol_execbench/core/data/solution.py`.
  - Example: `examples/miopen/softmax/`.
- Composable Kernel
  - Supported language category `ck` in `src/sol_execbench/core/data/solution.py`.
  - Example: `examples/ck/gemm/`.
  - Header availability is tested in `tests/conftest.py` through `/opt/rocm/include/ck/ck.hpp`.
- rocWMMA
  - Supported language category `rocwmma` in `src/sol_execbench/core/data/solution.py`.
  - Example: `examples/rocwmma/gemm/`.
  - Header availability is tested in `tests/conftest.py` through `/opt/rocm/include/rocwmma/rocwmma.hpp`.
- HIP C++
  - Supported language category `hip_cpp` in `src/sol_execbench/core/data/solution.py`.
  - Examples: `examples/hip_cpp/rmsnorm/` and `examples/hip_cpp/flux_rope/`.

## Datasets And External Benchmark Assets

- Downloaded benchmark assets belong under `data/`.
- FlashInfer Trace safetensors roots are integrated through `FLASHINFER_TRACE_DIR`.
- Safetensors loading/staging:
  - `src/sol_execbench/core/bench/io.py`
  - `src/sol_execbench/driver/problem_packager.py`
  - `scripts/run_dataset.py`
- Dataset migration and manifest workflows:
  - `src/sol_execbench/core/dataset/migration.py`
  - `src/sol_execbench/core/dataset/manifest.py`
  - `src/sol_execbench/core/dataset/inventory.py`
  - `scripts/download_solexecbench.py`
  - `scripts/download_data.sh`
- Dataset batch runner:
  - `scripts/run_dataset.py`
- Dataset package dependency:
  - `datasets` in `pyproject.toml`.
- Data redistribution checks:
  - `scripts/check_dataset_redistribution.py`.

## Benchmark User Code Integration

- User solutions are loaded from `solution.json` and staged by `src/sol_execbench/driver/problem_packager.py`.
- Python/Triton solutions are imported by the staged evaluator in `src/sol_execbench/driver/templates/eval_driver.py`.
- HIP/C++ solutions are compiled to `benchmark_kernel.so` by `src/sol_execbench/driver/templates/build_ext.py`.
- Entry points use the `{file_path}::{function_name}` format validated in `src/sol_execbench/core/data/solution.py`.
- Source paths are validated to reject absolute paths and parent traversal.
- Compile flags are validated to reject response files, host path injection, runtime linker manipulation, and unsafe external-path flags.
- Reward-hack and integrity checks live in `src/sol_execbench/core/bench/reward_hack.py` and are invoked by `src/sol_execbench/driver/templates/eval_driver.py`.

## Filesystem And Process Boundaries

- CLI evaluation executes a staged subprocess from `src/sol_execbench/cli/main.py`.
- Staging directories contain copied problem files, user sources, optional symlinked/copy safetensors blobs, `eval_driver.py`, and optionally `build_ext.py`.
- Evaluation subprocess timeout defaults are exposed by CLI options in `src/sol_execbench/cli/main.py`.
- No-trace subprocess failures emit bounded diagnostic sidecars instead of writing invalid canonical traces.
- `src/sol_execbench/core/bench/stderr.py` filters benign ROCm stderr before diagnostics are persisted.

## External Package Indexes And Supply Chain Inputs

- PyPI is the default uv index in `pyproject.toml`.
- PyTorch ROCm wheel indexes are explicit uv indexes in `pyproject.toml`.
- Docker copies uv from `ghcr.io/astral-sh/uv:0.11.18` in `docker/Dockerfile`.
- ROCm dev images come from the `rocm/dev-ubuntu-24.04` repository as declared in `docker/Dockerfile` and `docker/rocm-targets.json`.
- License and notice context is tracked in `LICENSE`, `THIRD_PARTY_NOTICES.txt`, and `docs/compliance.md`.

## GitHub And Release Workflow

- Contribution policy expects GitHub issues and DCO sign-off in `CONTRIBUTING.md` and `AGENTS.md`.
- Security policy is in `SECURITY.md`.
- Release/prerelease checks are local scripts, not a visible CI config in the repo:
  - `scripts/check_prerelease_readiness.py`
  - `scripts/build_prerelease_artifact_bundle.py`
  - `scripts/release_candidate_validation.py`
- Release notes and readiness docs are under `docs/` and `docs/releases/`.
