<!-- generated-by: gsd-doc-writer -->
# Configuration

SOL ExecBench ROCm Port has no required application environment variables in
the Python package itself. Runtime behavior is configured through CLI flags,
optional benchmark `config.json` files, package dependency configuration in
`pyproject.toml`, and Docker environment variables in `docker/Dockerfile`.

## Environment Variables

No `.env.example`, `.env.sample`, or source-level `process.env` configuration
exists in this Python project. The Docker image defines runtime environment
variables for ROCm and `uv`:

| Variable | Required | Default | Description |
| --- | --- | --- | --- |
| `ROCM_PATH` | Container-only | `/opt/rocm` | Path to the ROCm installation inside the Docker image. |
| `HIP_PATH` | Container-only | `/opt/rocm` | Path used by HIP tooling in the Docker image. |
| `HIP_PLATFORM` | Container-only | `amd` | Selects AMD HIP platform behavior inside the Docker image. |
| `PATH` | Container-only | Includes `/opt/rocm/bin`, `/opt/rocm/llvm/bin`, and `/venv/bin` | Makes ROCm tools and the project virtual environment available. |
| `LD_LIBRARY_PATH` | Container-only | `/opt/rocm/lib` | Makes ROCm shared libraries visible in the container. |
| `UV_CACHE_DIR` | Container-only | `/home/${HOST_USER}/.cache/uv` | Cache location for `uv` during Docker builds. |
| `UV_LINK_MODE` | Container-only | `copy` | Configures `uv` install behavior in the Docker build. |
| `UV_COMPILE_BYTECODE` | Container-only | `1` | Enables bytecode compilation during `uv` sync in the Docker build. |
| `UV_PYTHON_DOWNLOADS` | Container-only | `never` | Prevents Python downloads during Docker builds. |
| `UV_PROJECT_ENVIRONMENT` | Container-only | `/venv` | Installs the project environment into `/venv`. |
| `PYTHONPATH` | Container-only | `/sol-execbench/src` | Lets the mounted source tree override the installed package at runtime. |

The CLI also sets `PYTORCH_ALLOC_CONF=expandable_segments:True` for compilation
and evaluation subprocesses in `src/sol_execbench/cli/main.py`.

## Config File Format

Benchmark configuration is optional. The CLI accepts `--config <path>` and
loads it into `BenchmarkConfig` from
`src/sol_execbench/core/bench/config/benchmark_config.py`.

Minimal example:

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
| `warmup_runs` | `10` | Number of warmup executions before measured timing. |
| `iterations` | `50` | Number of measured iterations. |
| `lock_clocks` | `false` | Requires ROCm GPU clocks to be locked before benchmarking when enabled. |
| `benchmark_reference` | `true` | Measures reference implementation latency when enabled. |
| `seed` | `200` | Seed used by the generated evaluation driver. |

## Required Vs Optional Settings

All `BenchmarkConfig` fields are optional because defaults are defined in the
dataclass. The validation rules are:

| Setting | Requirement | Source |
| --- | --- | --- |
| `warmup_runs` | Must be greater than or equal to `0`. | `src/sol_execbench/core/bench/config/benchmark_config.py` |
| `iterations` | Must be greater than `0`. | `src/sol_execbench/core/bench/config/benchmark_config.py` |

The CLI input files are required at runtime. A user must provide either a
problem directory containing `definition.json` and `workload.jsonl`, or explicit
`--definition` and `--workload` paths. A solution must come from either
`--solution` or `solution.json` in the problem directory.

## Defaults

The benchmark defaults are:

| Setting | Default |
| --- | --- |
| `warmup_runs` | `10` |
| `iterations` | `50` |
| `lock_clocks` | `false` |
| `benchmark_reference` | `true` |
| `seed` | `200` |

The CLI defaults are:

| Option | Default |
| --- | --- |
| `--compile-timeout` | `120` seconds |
| `--timeout` | `600` seconds |
| `--json` | Disabled |
| `--lock-clocks` | Disabled |
| `--keep-staging` | Disabled |
| `--verbose` | Disabled |

## Per-Environment Overrides

No separate development, staging, or production configuration files are present.
For local runs, pass CLI flags or a benchmark `config.json`. For Docker runs,
use `./scripts/run_docker.sh` so the container receives ROCm device access,
repository mounts, and the runtime environment defined in `docker/Dockerfile`
and `docker/entrypoint.sh`.

## Dependency Configuration

`pyproject.toml` pins the package name, Python version range, console scripts,
runtime dependencies, development dependencies, Ruff exclusions, pytest
markers, and ROCm wheel indexes. The project requires Python `>=3.12,<3.14`.
On Linux and Windows, `torch` and `torchvision` resolve from the
`pytorch-rocm71` index, and `triton-rocm` resolves from the PyTorch ROCm root
index on Linux.

