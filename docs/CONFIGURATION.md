<!-- generated-by: gsd-doc-writer -->
# Configuration

SOL ExecBench ROCm Port is configured through CLI flags, optional benchmark
configuration JSON, `pyproject.toml`, Docker target metadata, Docker wrapper
inputs, and runtime environment variables. The repository does not contain a
required application-level `.env` file.

## Environment Variables

No environment variable is required for normal host CLI startup. The variables
below are optional runtime, Docker, diagnostic, or build inputs discovered in
`src/`, `scripts/`, and `docker/`.

| Variable | Required | Default | Description |
| --- | --- | --- | --- |
| `PYTORCH_ALLOC_CONF` | Optional | `expandable_segments:True` in compile/eval subprocesses | Set by the CLI subprocess launcher for staged PyTorch ROCm compilation and evaluation. |
| `PYTORCH_ROCM_ARCH` | Optional | Derived from solution `target_hardware` when unset | Overrides the ROCm architecture list used by PyTorch extension builds. |
| `SOLEXECBENCH_ENV_SNAPSHOT` | Optional | Unset | Set to `1` to write an environment snapshot sidecar next to `--output`. |
| `SOLEXECBENCH_ENV_SNAPSHOT_PATH` | Optional | Unset | Explicit environment snapshot sidecar output path. |
| `HIP_VISIBLE_DEVICES` | Optional | Unset | Device visibility filter recorded in environment and runtime evidence. |
| `ROCR_VISIBLE_DEVICES` | Optional | Unset | ROCr device visibility filter recorded in environment and runtime evidence. |
| `HSA_OVERRIDE_GFX_VERSION` | Optional | Unset | Forced HSA architecture override recorded in environment snapshots. |
| `CUDA_VISIBLE_DEVICES` | Optional | Unset | Compatibility visibility variable recorded by Docker runtime evidence when present. |
| `GPU_DEVICE_ORDINAL` | Optional | Unset | Compatibility visibility variable recorded by Docker runtime evidence when present. |
| `SOL_EXECBENCH_CLOCKS_LOCKED` | Optional | `0` when unset | Set by `docker/entrypoint.sh` after clock-lock attempts; read by clock-lock checks. |
| `SOL_EXECBENCH_SCLK_LEVEL` | Optional | Device preset when available | Overrides the SCLK DPM level used by the ROCm clock-lock helper. |
| `SOL_EXECBENCH_MCLK_LEVEL` | Optional | Device preset when available | Overrides the MCLK DPM level used by the ROCm clock-lock helper. |
| `FLASHINFER_TRACE_DIR` | Optional | `/sol-execbench/data/flashinfer-trace` under `scripts/run_docker.sh` | Adds the FlashInfer trace safetensors lookup root for evaluation. |
| `IMAGE_NAME` | Optional | `sol-execbench` | Docker wrapper local image name. |
| `IMAGE_TAG` | Optional | `rocm-<selected Docker tag>` | Docker wrapper local image tag. The default target resolves to `rocm-7.1.1-complete`. |
| `ROCM_DOCKER_IMAGE` | Optional | `rocm/dev-ubuntu-24.04` for unknown-target override | Docker image repository override when `--allow-unknown-target` is used. |
| `ROCM_DOCKER_TAG` | Optional | Selected target ID for unknown-target override | Docker image tag override when `--allow-unknown-target` is used. |
| `SOL_EXECBENCH_ALLOW_MIXED_VERSION_DEPENDENCIES` | Optional | `0` | Allows dependency probe diagnostics for mixed-version stacks. |
| `SOL_EXECBENCH_ALLOW_UNTESTED_TARGET_SMOKE` | Optional | `0` | Allows `not_tested` targets to run smoke/E2E commands without validation claims. |
| `SOL_EXECBENCH_RECORD_CONTAINER_VALIDATION` | Optional | `0` | Records successful target-container wrapper benchmark evidence as `container_validated`. |
| `SOL_EXECBENCH_HOST_PYTHON` | Optional | `uv run python` | Host Python executable override for Docker wrapper helper commands. |
| `SOL_EXECBENCH_COMPATIBILITY_ENTRY` | Optional | Unset | Per-target compatibility JSON sidecar path. |
| `SOL_EXECBENCH_COMPATIBILITY_MATRIX` | Optional | Unset | Aggregate compatibility matrix JSON path. |
| `SOL_EXECBENCH_RUN_DOCKER_DRY_RUN` | Optional | `0` | Enables dry-run behavior in `scripts/run_docker.sh`. |
| `SOL_EXECBENCH_DOCKER_CONTEXT` | Optional | `docker context show` output | Test/debug override for Docker context preflight evidence. |
| `SOL_EXECBENCH_DOCKER_HOST` | Optional | `docker context inspect` output | Test/debug override for Docker host preflight evidence. |
| `SOL_EXECBENCH_DEV_KFD_PRESENT` | Optional | Filesystem probe of `/dev/kfd` | Test/debug override for Docker runtime preflight evidence. |
| `SOL_EXECBENCH_DEV_KFD_ACCESSIBLE` | Optional | Read/write probe of `/dev/kfd` | Test/debug override for Docker runtime preflight evidence. |
| `SOL_EXECBENCH_DEV_DRI_PRESENT` | Optional | Filesystem probe of `/dev/dri` | Test/debug override for Docker runtime preflight evidence. |
| `SOL_EXECBENCH_DEV_DRI_ACCESSIBLE` | Optional | Render/card device access probe under `/dev/dri` | Test/debug override for Docker runtime preflight evidence. |
| `SOL_EXECBENCH_GPU_ACCESSIBLE` | Optional | Unset | Test/debug override for Docker runtime preflight GPU accessibility. |
| `SOL_EXECBENCH_HOST_ROCM_VERSION` | Optional | Unset | Runtime evidence override for host ROCm version. |
| `SOL_EXECBENCH_HOST_DRIVER_VERSION` | Optional | Unset | Runtime evidence override for host driver version. |
| `SOL_EXECBENCH_IMAGE_DIGEST` | Optional | Unset | Runtime evidence override for container image digest. |
| `SOL_EXECBENCH_RUNTIME_DEVICE_COUNT` | Optional | Unset | Runtime evidence override for device count. |
| `SOL_EXECBENCH_RUNTIME_DEVICE_NAME` | Optional | Unset | Runtime evidence override for device name. |
| `SOL_EXECBENCH_RUNTIME_GFX_ARCHITECTURE` | Optional | Unset | Runtime evidence override for gfx architecture. |
| `SOL_EXECBENCH_GPU_CLK_MHZ` | Optional | Empty string in Docker wrapper environment | Forwarded into Docker runs for GPU clock diagnostics. |
| `SOL_EXECBENCH_DRAM_CLK_MHZ` | Optional | Empty string in Docker wrapper environment | Forwarded into Docker runs for DRAM clock diagnostics. |
| `SOL_EXECBENCH_DEPENDENCY_TORCH_DISTRIBUTION_VERSION` | Optional | Unset | Dependency preflight/runtime evidence override for installed Torch distribution version. |
| `SOL_EXECBENCH_DEPENDENCY_TORCH_VERSION` | Optional | Unset | Dependency preflight/runtime evidence override for imported Torch version. |
| `SOL_EXECBENCH_DEPENDENCY_TORCH_LOCAL_VERSION` | Optional | Unset | Dependency preflight/runtime evidence override for Torch local version suffix. |
| `SOL_EXECBENCH_DEPENDENCY_TORCH_ROCM_TARGET` | Optional | Unset | Dependency preflight/runtime evidence override for expected ROCm wheel target. |
| `SOL_EXECBENCH_DEPENDENCY_TORCH_HIP_VERSION` | Optional | Unset | Dependency preflight/runtime evidence override for `torch.version.hip`. |
| `SOL_EXECBENCH_DEPENDENCY_TORCH_CUDA_VERSION` | Optional | Unset | Dependency preflight/runtime evidence override for `torch.version.cuda`. |
| `SOL_EXECBENCH_DEPENDENCY_TORCH_DEVICE_AVAILABLE` | Optional | Unset | Dependency preflight/runtime evidence override for Torch device availability. |
| `SOL_EXECBENCH_DEPENDENCY_TORCH_IMPORT_ERROR` | Optional | Unset | Dependency preflight/runtime evidence override for Torch import failure text. |
| `SOL_EXECBENCH_DEPENDENCY_TORCHVISION_DISTRIBUTION_VERSION` | Optional | Unset | Dependency preflight/runtime evidence override for installed torchvision distribution version. |
| `SOL_EXECBENCH_DEPENDENCY_TRITON_ROCM_DISTRIBUTION_VERSION` | Optional | Unset | Dependency preflight/runtime evidence override for installed `triton-rocm` distribution version. |
| `SOL_EXECBENCH_DEPENDENCY_TRITON_ROCM_STATUS` | Optional | Unset | Dependency preflight/runtime evidence override for Triton ROCm status. |
| `SOL_EXECBENCH_DEPENDENCY_CONTAINER_ROCM_USER_SPACE_VERSION` | Optional | Unset | Dependency preflight/runtime evidence override for container ROCm user-space version. |
| `SOL_EXECBENCH_DEPENDENCY_HIPCC_VERSION` | Optional | Unset | Dependency preflight/runtime evidence override for `hipcc` version. |
| `SOL_EXECBENCH_DEPENDENCY_TOOLCHAIN_ROCM_VERSION` | Optional | Unset | Dependency preflight/runtime evidence override for toolchain ROCm version. |
| `ROCM_PATH` | Optional | `/opt/rocm` in Docker image | ROCm installation root in `docker/Dockerfile`. |
| `HIP_PATH` | Optional | `/opt/rocm` in Docker image | HIP installation root in `docker/Dockerfile`. |
| `HIP_PLATFORM` | Optional | `amd` in Docker image | HIP platform selector in `docker/Dockerfile`. |
| `UV_CACHE_DIR` | Optional | `/home/${HOST_USER}/.cache/uv` in Docker image | UV cache directory in `docker/Dockerfile`. |
| `UV_LINK_MODE` | Optional | `copy` in Docker image | UV link behavior in `docker/Dockerfile`. |
| `UV_COMPILE_BYTECODE` | Optional | `1` in Docker image | Enables bytecode compilation in Docker installs. |
| `UV_PYTHON_DOWNLOADS` | Optional | `never` in Docker image | Disables Python downloads during Docker installs. |
| `UV_PROJECT_ENVIRONMENT` | Optional | `/venv` in Docker image | Docker image virtual environment path. |
| `HOST_UID` | Optional | `1000` Docker build argument | Host user ID used when creating the Docker image user. |
| `HOST_GID` | Optional | `1000` Docker build argument | Host group ID used when creating the Docker image group. |
| `HOST_USER` | Optional | `sol-execbench` Docker build argument | Host user name used when creating the Docker image user. |
| `PYTORCH_TORCH_VERSION` | Optional | `2.10.0+rocm7.1` Docker build argument | Torch wheel version installed in the Docker image. |
| `PYTORCH_TORCHVISION_VERSION` | Optional | `0.25.0+rocm7.1` Docker build argument | Torchvision wheel version installed in the Docker image. |
| `PYTORCH_ROCM_INDEX_URL` | Optional | `https://download.pytorch.org/whl/rocm7.1` Docker build argument | PyTorch ROCm wheel index URL. |
| `TRITON_ROCM_VERSION` | Optional | `3.6.0` Docker build argument | `triton-rocm` wheel version installed in the Docker image. |
| `TRITON_ROCM_INDEX_URL` | Optional | `https://download.pytorch.org/whl/` Docker build argument | Extra wheel index URL used for `triton-rocm`. |

## Config File Format

The benchmark evaluator accepts an optional JSON config file through
`--config`. It is loaded into `BenchmarkConfig` from
`src/sol_execbench/core/bench/config/benchmark_config.py`.

```json
{
  "warmup_runs": 10,
  "iterations": 50,
  "lock_clocks": false,
  "benchmark_reference": true,
  "seed": 200
}
```

| Field | Default | Description |
| --- | --- | --- |
| `warmup_runs` | `10` | Number of warmup runs before measurement. Must be greater than or equal to `0`. |
| `iterations` | `50` | Number of timed iterations. Must be greater than `0`. |
| `lock_clocks` | `false` | Whether evaluation requires clocks to be locked. |
| `benchmark_reference` | `true` | Whether to benchmark the reference implementation. |
| `seed` | `200` | Integer seed for benchmark input generation. |

The Docker target manifest at `docker/rocm-targets.json` is another repository
configuration file. It declares `default_target_id`,
`requested_rocm_user_space_version`, Docker image tags, PyTorch ROCm wheel
policies, and Triton ROCm wheel policies for supported container targets.

`provenance.toml` is the machine-readable source attribution policy. It
classifies active files as upstream-retained, derivative-modified, independent
ROCm work, or generated/planning material; defines header policy for each
class; and lists files that may retain NVIDIA SPDX notices. The prerelease
readiness gate consumes this manifest with `docs/provenance.md`.

## Required vs Optional Settings

There are no required environment variables for starting the CLI. Required
inputs are passed as files or positional arguments:

- `sol-execbench <problem_dir>` requires `definition.json` and `workload.jsonl`
  in the problem directory.
- If no problem directory is used, `--definition` and `--workload` are required.
- A solution must be supplied through `--solution` or by a conventional
  `solution.json` in the problem directory.
- `contract`, `doctor`, and `toolchain` subcommands only support `--json`
  output and raise a Click exception without it.
- `BenchmarkConfig.warmup_runs` must be `>= 0`; `iterations` must be `> 0`.
- `scripts/run_docker.sh --target`, `--compatibility-entry`, and
  `--compatibility-matrix` require argument values when those flags are used.

Clock-lock settings are optional, but `--lock-clocks` changes behavior by
forcing `BenchmarkConfig.lock_clocks = true`. If clocks are not locked,
clock-sensitive evaluation paths can reject the run based on
`SOL_EXECBENCH_CLOCKS_LOCKED`.

## Defaults

| Setting | Default | Source |
| --- | --- | --- |
| Benchmark warmup runs | `10` | `BenchmarkConfig.warmup_runs` |
| Benchmark iterations | `50` | `BenchmarkConfig.iterations` |
| Benchmark clock-lock requirement | `false` | `BenchmarkConfig.lock_clocks` |
| Benchmark reference timing | `true` | `BenchmarkConfig.benchmark_reference` |
| Benchmark seed | `200` | `BenchmarkConfig.seed` |
| CLI compile timeout | `120` seconds | `--compile-timeout` option |
| CLI evaluation timeout | `600` seconds | `--timeout` option |
| CLI profiling mode | `none` | `--profile` option |
| CLI static evidence mode | `none` | `--static-evidence` option |
| Docker target | `rocm-7.1.1-ubuntu-24.04-container` | `docker/rocm-targets.json` |
| Docker base image | `rocm/dev-ubuntu-24.04:7.1.1-complete` | Default Docker target and `docker/Dockerfile` |
| Docker local image name | `sol-execbench` | `scripts/run_docker.sh` |
| Docker local image tag | `rocm-<selected Docker tag>` | `scripts/run_docker.sh` |
| Docker FlashInfer trace root | `/sol-execbench/data/flashinfer-trace` | `scripts/run_docker.sh` |
| Python package version | `1.0.2` | `pyproject.toml`; separate from the v1.26 research-prerelease milestone tag |
| Python requirement | `>=3.12,<3.14` | `pyproject.toml` |
| Default ROCm Torch wheel | `torch==2.10.0+rocm7.1` | `pyproject.toml` Linux x86_64 marker and Docker target metadata |
| Default ROCm torchvision wheel | `torchvision==0.25.0+rocm7.1` | `pyproject.toml` Linux x86_64 marker and Docker target metadata |
| Default Triton ROCm wheel | `triton-rocm==3.6.0` | `pyproject.toml` Linux x86_64 marker and Docker target metadata |
| Provenance policy document | `docs/provenance.md` | `provenance.toml` |

## Per-Environment Overrides

The repository does not define `.env.development`, `.env.production`, or
`.env.test` files, and it does not contain deployment-specific configuration
files. Use these source-backed override paths instead:

- Host benchmark runs: pass CLI flags such as `--config`, `--compile-timeout`,
  `--timeout`, `--output`, `--profile`, and `--static-evidence`.
- Per-problem benchmark settings: place a `config.json` next to
  `definition.json` and `workload.jsonl`, or pass a config file explicitly with
  `--config`.
- Docker ROCm stack selection: use `./scripts/run_docker.sh --target <id>` for
  declared targets in `docker/rocm-targets.json`.
- Docker image overrides for unknown targets: use `--allow-unknown-target` with
  `ROCM_DOCKER_IMAGE` and `ROCM_DOCKER_TAG`.
- Diagnostic and CI-style preflight overrides: set the
  `SOL_EXECBENCH_DEPENDENCY_*`, `SOL_EXECBENCH_RUNTIME_*`,
  `SOL_EXECBENCH_DEV_*`, `SOL_EXECBENCH_DOCKER_*`, and compatibility sidecar
  variables listed above.
- Source attribution policy: update `provenance.toml` and
  `docs/provenance.md`; prior blanket header corrections are handled as
  ordinary commits unless a separate legal review requires history rewriting.

## CLI Flags

Primary evaluator form:

```bash
uv run sol-execbench <problem_dir> --solution solution.json
```

Explicit input form:

```bash
uv run sol-execbench \
  --definition definition.json \
  --workload workload.jsonl \
  --solution solution.json
```

| Flag | Default | Purpose |
| --- | --- | --- |
| `--definition` | None | Path to `definition.json` when not using a problem directory. |
| `--workload` | None | Path to `workload.jsonl` when not using a problem directory. |
| `--solution` | Conventional `solution.json` in problem directory when present | Path to solution JSON. |
| `--config` | None | Optional benchmark config JSON. |
| `--compile-timeout` | `120` | HIP/C++ compilation timeout in seconds. |
| `--timeout` | `600` | Evaluation subprocess timeout in seconds. |
| `-o`, `--output` | None | Trace JSONL output path. |
| `--json` | Disabled | Print trace JSON lines to stdout. |
| `--lock-clocks` | Disabled | Require GPU clocks to be locked. |
| `--keep-staging` | Disabled | Preserve the temporary staging directory. |
| `--profile` | `none` | Use `rocprofv3` for optional diagnostic profiling when set to `rocprofv3`. |
| `--static-evidence` | `none` | Collect optional diagnostic static kernel evidence when set to `auto`. |
| `-v`, `--verbose` | Disabled | Show subprocess output and staging details. |

No-trace diagnostic sidecars are not controlled by a separate flag. When an
evaluation subprocess produces no parseable trace JSONL, the CLI writes a
bounded diagnostic-only sidecar next to `--output`, in the kept staging
directory, or in the system temp directory depending on the available path.
That sidecar records stdout/stderr tails and line counts and is not canonical
trace JSONL.

Metadata and diagnostic subcommands require `--json`:

```bash
uv run sol-execbench contract --json
uv run sol-execbench doctor --json
uv run sol-execbench toolchain --json
```

`toolchain` also accepts `--evidence-level`, `--artifact-type`, `--gpu-arch`,
`--hardware-generation`, `--rocm-version`, and `--list-registry`.

Local dataset migration subcommands live under the same `sol-execbench` entry
point:

```bash
uv run sol-execbench dataset migrate-sol <source_root> <output_root> \
  --category L1 --manifest out/sol-migration-manifest.json --json
uv run sol-execbench dataset migrate-flashinfer <source_root> <output_root> \
  --manifest out/flashinfer-migration-manifest.json --json
```

| Subcommand | Option | Purpose |
| --- | --- | --- |
| `dataset migrate-sol` | `--category` | Restrict SOL-ExecBench migration to one or more categories. |
| `dataset migrate-sol` | `--source-revision` | Record the source dataset revision or local commit ref in the manifest. |
| `dataset migrate-sol` | `--manifest` | Write the migration manifest to an explicit path. |
| `dataset migrate-sol` | `--json` | Print the migration manifest JSON to stdout. |
| `dataset migrate-flashinfer` | `--source-revision` | Record the FlashInfer Trace source revision or local commit ref in the manifest. |
| `dataset migrate-flashinfer` | `--manifest` | Write the migration manifest to an explicit path. |
| `dataset migrate-flashinfer` | `--json` | Print the migration manifest JSON to stdout. |

## Package Configuration

`pyproject.toml` defines:

- Package name `sol-execbench`
- Version `1.0.2`, which is the Python package version rather than the v1.26
  research-prerelease milestone tag
- Python range `>=3.12,<3.14`
- Console scripts `sol-execbench` and `sol-execbench-baseline`
- Runtime dependencies, including PyTorch ROCm, torchvision ROCm, `triton-rocm`,
  Pydantic, Click, Rich, datasets, and native build helpers
- Development dependencies for pytest, pytest-xdist, Ruff, Ty, and pre-commit
- Pytest markers
- Ruff exclusions
- Ty source roots
- UV package indexes for PyPI, PyTorch ROCm 7.1, and the PyTorch ROCm package root

## Docker Wrapper Settings

`scripts/run_docker.sh` supports:

- `--build` to build the selected Docker image.
- `--target <id>` to select a declared target from `docker/rocm-targets.json`.
- `--allow-unknown-target` with `ROCM_DOCKER_IMAGE` and `ROCM_DOCKER_TAG`.
- `--allow-mixed-version-dependencies` for mixed-version dependency diagnostics.
- `--allow-untested-target-smoke` for non-authoritative smoke/E2E runs on
  `not_tested` targets.
- `--record-container-validation` to write container validation evidence after
  successful wrapper execution.
- `--preflight-only` to run preflight classification without launching the
  benchmark command.
- `--compatibility-entry <path>` and `--compatibility-matrix <path>` to write
  compatibility sidecars.

The wrapper derives PyTorch and Triton Docker build arguments from
`docker/rocm-targets.json` so ROCm 7.0, 7.1, and 7.2 images can install
target-specific ROCm wheel stacks without changing the project lockfile.

## Dataset Runner Options

`scripts/run_dataset.py` accepts dataset-scale options outside the main
`sol-execbench` Click command:

| Flag | Purpose |
| --- | --- |
| `--ready-subset` | Bound execution to a previously generated ready-subset sidecar. |
| `--readiness` | Enrich execution closure with readiness blocker records. |
| `--execution-closure` | Write an execution-closure JSON sidecar; defaults under `--output` when `--ready-subset` is supplied. |
| `--dataset-manifest` | Include dataset manifest provenance in closure checks. |
| `--ready-subset` plus `--readiness` | Preserve ready-subset denominators, readiness classes, blocker codes, and exclusion reasons in closure records. |
| `--phase all|traces|derived|timing` | Select the dataset pass. `all` preserves normal execution, `traces` runs GPU trace collection, `derived` rebuilds reports from existing traces, and `timing` collects profiler-backed timing evidence from existing traces. |
| `--jobs <N>|auto` | Use parallel workers for safe CPU/I/O-only phases. Values greater than `1` are honored only with `--phase derived`; GPU and profiler phases remain serial. |
| `--rerun` | Re-evaluate existing traces instead of reusing passing results. |
| `--amd-score-report` | Write an opt-in AMD-native derived score report. |
| `--amd-sol-bound-dir` | Materialize AMD SOL bound sidecars used by score reporting. |
| `--solar-derivation` | Materialize SOLAR derivation sidecars. |
| `--timing-evidence-dir` | Write source-specific ROCm timing evidence sidecars. |

Existing traces are reused only when they exist, have no failed workloads,
`--rerun` is not set, and any requested execution-closure provenance matches
the current manifest, ready subset, readiness, solution, and derived-evidence
requirements.

The `auto` job count uses the current CPU count capped by the number of selected
problems and an internal upper bound. Derived sidecar names are scoped by
problem path during dataset runs, so two problems with the same definition name
and workload UUID do not overwrite each other's AMD SOL or SOLAR derivation
files.

## Docker Image Settings

`docker/Dockerfile` sets ROCm and `uv` environment defaults:

| Variable | Default |
| --- | --- |
| `ROCM_PATH` | `/opt/rocm` |
| `HIP_PATH` | `/opt/rocm` |
| `HIP_PLATFORM` | `amd` |
| `PATH` | Includes `/opt/rocm/bin`, `/opt/rocm/llvm/bin`, and `/venv/bin`. |
| `LD_LIBRARY_PATH` | `/opt/rocm/lib` |
| `HOME` | `/home/${HOST_USER}` |
| `UV_CACHE_DIR` | `/home/${HOST_USER}/.cache/uv` |
| `UV_LINK_MODE` | `copy` |
| `UV_COMPILE_BYTECODE` | `1` |
| `UV_PYTHON_DOWNLOADS` | `never` |
| `UV_PROJECT_ENVIRONMENT` | `/venv` |
| `PYTHONPATH` | `/sol-execbench/src` |

Docker build arguments include `ROCM_DOCKER_IMAGE`, `ROCM_DOCKER_TAG`,
`PYTORCH_TORCH_VERSION`, `PYTORCH_TORCHVISION_VERSION`,
`PYTORCH_ROCM_INDEX_URL`, `TRITON_ROCM_VERSION`, `TRITON_ROCM_INDEX_URL`,
`HOST_UID`, `HOST_GID`, and `HOST_USER`.

## Optional Evidence Outputs

These settings create sidecars without changing trace JSONL schema:

- `--profile rocprofv3` writes profiler metadata next to `--output`, or under
  the staging directory when no output file is provided.
- `--static-evidence auto` writes static kernel evidence metadata for native
  solution builds.
- `SOLEXECBENCH_ENV_SNAPSHOT=1` writes an environment sidecar next to
  `--output`.
- `SOLEXECBENCH_ENV_SNAPSHOT_PATH=<path>` writes an environment sidecar to an
  explicit path.

## Configuration Boundaries

No deployment, staging, or production config files are present. For local
benchmark changes, use CLI flags or benchmark config JSON. For ROCm container
runs, use `./scripts/run_docker.sh` so target selection, device mounting,
dependency preflight, and compatibility sidecars stay aligned with repository
logic.
