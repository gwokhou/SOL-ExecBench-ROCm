<!-- generated-by: gsd-doc-writer -->
# Configuration

SOL ExecBench ROCm Port is configured through CLI flags, optional benchmark
configuration JSON, `pyproject.toml`, Docker wrapper inputs, and runtime
environment variables. There is no required application-level `.env` file.

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

Metadata and diagnostic subcommands require `--json`:

```bash
uv run sol-execbench contract --json
uv run sol-execbench doctor --json
uv run sol-execbench toolchain --json
```

`toolchain` also accepts `--evidence-level`, `--artifact-type`, `--gpu-arch`,
`--hardware-generation`, `--rocm-version`, and `--list-registry`.

## Benchmark Config JSON

The optional `--config` file loads into `BenchmarkConfig` from
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

| Field | Default | Validation |
| --- | --- | --- |
| `warmup_runs` | `10` | Must be greater than or equal to `0`. |
| `iterations` | `50` | Must be greater than `0`. |
| `lock_clocks` | `false` | Boolean. |
| `benchmark_reference` | `true` | Boolean. |
| `seed` | `200` | Integer. |

The `--lock-clocks` CLI flag overrides the loaded config by setting
`lock_clocks` to `true`.

## Package Configuration

`pyproject.toml` defines:

- Package name `sol-execbench`
- Version `1.0.2`
- Python range `>=3.12,<3.14`
- Console scripts `sol-execbench` and `sol-execbench-baseline`
- Runtime dependencies, including PyTorch ROCm, torchvision ROCm, `triton-rocm`,
  Pydantic, Click, Rich, datasets, and native build helpers
- Development dependencies for pytest, pytest-xdist, Ruff, and Ty
- Pytest markers
- Ruff exclusions
- Ty source roots
- UV package indexes for PyPI, PyTorch ROCm 7.1, and the PyTorch ROCm package root

## Runtime Environment Variables

| Variable | Default | Used By | Purpose |
| --- | --- | --- | --- |
| `PYTORCH_ALLOC_CONF` | `expandable_segments:True` in compile/eval subprocesses | CLI subprocess launch | Sets PyTorch allocator behavior for staged compilation and evaluation. |
| `PYTORCH_ROCM_ARCH` | Derived from solution targets when unset | Native build template and PyTorch extension build | Overrides ROCm architecture list for native builds. |
| `SOLEXECBENCH_ENV_SNAPSHOT` | Unset | CLI | Set to `1` to write an environment snapshot next to `--output`. |
| `SOLEXECBENCH_ENV_SNAPSHOT_PATH` | Unset | CLI | Explicit environment snapshot output path. |
| `HIP_VISIBLE_DEVICES` | Unset | Environment snapshots and runtime evidence | Records HIP-visible device filtering. |
| `ROCR_VISIBLE_DEVICES` | Unset | Environment snapshots and runtime evidence | Records ROCr-visible device filtering. |
| `HSA_OVERRIDE_GFX_VERSION` | Unset | Environment snapshots | Records forced HSA architecture overrides. |
| `SOL_EXECBENCH_CLOCKS_LOCKED` | `0` when unset | Clock-lock checks and Docker entrypoint | Records whether GPU clocks were locked by the entrypoint or helper. |
| `SOL_EXECBENCH_SCLK_LEVEL` | Device preset when available | Clock-lock helper | Overrides selected SCLK DPM level. |
| `SOL_EXECBENCH_MCLK_LEVEL` | Device preset when available | Clock-lock helper | Overrides selected MCLK DPM level. |
| `FLASHINFER_TRACE_DIR` | `/sol-execbench/data/flashinfer-trace` under wrapper | Evaluation driver and Docker wrapper | Adds FlashInfer trace safetensors lookup root. |

## Docker Wrapper Settings

`scripts/run_docker.sh` reads these user-facing variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `IMAGE_NAME` | `sol-execbench` | Local Docker image name. |
| `IMAGE_TAG` | `rocm-${selected_docker_tag}` | Local Docker image tag. The default declared target currently builds `sol-execbench:rocm-7.1.1-complete`; set `IMAGE_TAG` to override. |
| `ROCM_DOCKER_IMAGE` | `rocm/dev-ubuntu-24.04` for unknown-target override | Docker image repository override when unknown targets are explicitly allowed. |
| `ROCM_DOCKER_TAG` | Selected target ID for unknown-target override | Docker image tag override when unknown targets are explicitly allowed. |
| `SOL_EXECBENCH_ALLOW_MIXED_VERSION_DEPENDENCIES` | `0` | Allows dependency probe diagnostics for mixed-version stacks. |
| `SOL_EXECBENCH_ALLOW_UNTESTED_TARGET_SMOKE` | `0` | Allows `not_tested` Targets to run smoke/E2E commands while preserving non-authoritative compatibility claims. |
| `SOL_EXECBENCH_COMPATIBILITY_ENTRY` | Unset | Optional per-target compatibility JSON sidecar path. |
| `SOL_EXECBENCH_COMPATIBILITY_MATRIX` | Unset | Optional aggregate compatibility matrix JSON path. |
| `SOL_EXECBENCH_RUN_DOCKER_DRY_RUN` | `0` | Enables dry-run behavior used by tests and diagnostics. |

The wrapper also supports many `SOL_EXECBENCH_DEPENDENCY_*` and
`SOL_EXECBENCH_RUNTIME_*` override variables for dependency preflight,
compatibility sidecars, and tests. These are diagnostic inputs, not normal
benchmark configuration.

## Docker Image Settings

`docker/Dockerfile` sets ROCm and `uv` environment defaults:

| Variable | Default |
| --- | --- |
| `ROCM_PATH` | `/opt/rocm` |
| `HIP_PATH` | `/opt/rocm` |
| `HIP_PLATFORM` | `amd` |
| `PATH` | Includes `/opt/rocm/bin`, `/opt/rocm/llvm/bin`, and `/venv/bin`. |
| `LD_LIBRARY_PATH` | `/opt/rocm/lib` |
| `UV_CACHE_DIR` | `/home/${HOST_USER}/.cache/uv` |
| `UV_LINK_MODE` | `copy` |
| `UV_COMPILE_BYTECODE` | `1` |
| `UV_PYTHON_DOWNLOADS` | `never` |
| `UV_PROJECT_ENVIRONMENT` | `/venv` |
| `PYTHONPATH` | `/sol-execbench/src` |

Docker build arguments include `ROCM_DOCKER_IMAGE`, `ROCM_DOCKER_TAG`,
`PYTORCH_TORCH_VERSION`, `PYTORCH_TORCHVISION_VERSION`,
`PYTORCH_ROCM_INDEX_URL`, `TRITON_ROCM_VERSION`, `TRITON_ROCM_INDEX_URL`,
`HOST_UID`, `HOST_GID`, and `HOST_USER`. The wrapper derives the PyTorch and
Triton build arguments from `docker/rocm-targets.json` so ROCm 7.0, 7.1, and
7.2 images can install target-specific ROCm wheel stacks without changing the
project lockfile.

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
