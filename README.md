<!-- generated-by: gsd-doc-writer -->
# SOL ExecBench ROCm Port

SOL ExecBench ROCm Port is a ROCm-only Python benchmark package for evaluating
LLM-generated GPU kernels on AMD hardware. It keeps the original SOL ExecBench
benchmark shape where practical, while replacing CUDA/NVIDIA execution paths
with ROCm, HIP, Triton ROCm, and AMD-native scoring and evidence helpers.

## Requirements

- Python `>=3.12,<3.14`
- `uv` for dependency and environment management
- Linux with ROCm-capable AMD hardware for GPU evaluation
- ROCm device access through `/dev/kfd` and `/dev/dri`
- ROCm 7.x user-space tooling for native HIP/C++ and profiler paths
- Docker, when using the provided ROCm container workflow

The project dependency configuration resolves PyTorch `2.10.0+rocm7.1` and
torchvision `0.25.0+rocm7.1` on Linux and Windows. On Linux it also resolves
`triton-rocm==3.6.0`. The Docker target manifest records ROCm 7.0.2, 7.1.1,
and 7.2.0 container targets for compatibility and evidence workflows.

## Installation

Install runtime and development dependencies:

```bash
uv sync --all-groups
```

Build and enter the ROCm Docker environment:

```bash
./scripts/run_docker.sh --build
```

Optional benchmark assets belong under `data/` and can be downloaded with:

```bash
uv run --with "huggingface-hub[cli]" ./scripts/download_data.sh
```

## Quick Start

Run a small included PyTorch problem:

```bash
uv run sol-execbench examples/pytorch/gemma3_swiglu \
  --solution examples/pytorch/gemma3_swiglu/solution_python.json
```

Run a Triton ROCm sample:

```bash
uv run sol-execbench examples/triton/rmsnorm \
  --solution examples/triton/rmsnorm/solution_triton.json
```

Write trace JSONL for later comparison:

```bash
uv run sol-execbench tests/sol_execbench/samples/linear_backward \
  --solution tests/sol_execbench/samples/linear_backward/solution_python.json \
  --output out/linear_backward.trace.jsonl
```

Run a small downloaded dataset batch:

```bash
uv run scripts/run_dataset.py data/SOL-ExecBench/benchmark --limit 5
```

## Usage Examples

Evaluate a problem directory containing `definition.json` and `workload.jsonl`:

```bash
uv run sol-execbench <problem_dir> --solution solution.json
```

Or pass the problem files explicitly:

```bash
uv run sol-execbench \
  --definition definition.json \
  --workload workload.jsonl \
  --solution solution.json
```

Common evaluator options:

| Flag | Purpose |
| --- | --- |
| `--config` | Load optional benchmark configuration JSON. |
| `--compile-timeout` | Set HIP/C++ compilation timeout in seconds. |
| `--timeout` | Set evaluation subprocess timeout in seconds. |
| `-o`, `--output` | Write trace JSONL to a file. |
| `--json` | Print trace JSON lines to stdout. |
| `--lock-clocks` | Require GPU clocks to be locked before benchmarking. |
| `--keep-staging` | Preserve the temporary staging directory. |
| `--profile rocprofv3` | Collect optional diagnostic `rocprofv3` profile artifacts. |
| `--static-evidence auto` | Collect optional diagnostic static kernel evidence for native builds. |
| `-v`, `--verbose` | Show subprocess output and staging details. |

Metadata and diagnostic subcommands:

```bash
uv run sol-execbench contract --json
uv run sol-execbench doctor --json
uv run sol-execbench toolchain --json
uv run sol-execbench toolchain --json --list-registry
```

Compare trace JSONL against one or more baselines:

```bash
uv run sol-execbench-baseline \
  --candidate out/candidate.trace.jsonl \
  --baseline out/baseline.trace.jsonl \
  --format text
```

Baseline comparison accepts repeated `--baseline` inputs, `--format text|json`,
`--output`, `--win-pct`, `--parity-pct`, and `--amd-native-claim` for guarded
AMD-native reporting.

## Supported Solution Categories

The ROCm schema accepts Python/Triton categories and native ROCm categories:

| Category | Notes |
| --- | --- |
| `pytorch` | Python implementation using PyTorch operators. |
| `triton` | Triton ROCm implementation. |
| `hip_cpp` | Native HIP/C++ implementation built through PyTorch extensions. |
| `hipblas` | Native ROCm implementation backed by hipBLAS. |
| `miopen` | Native ROCm implementation backed by MIOpen. |
| `ck` | Native ROCm implementation using Composable Kernel. |
| `rocwmma` | Native ROCm implementation using rocWMMA. |

Legacy CUDA/NVIDIA schema values such as `cuda_cpp`, `cublas`, `cudnn`,
`cutlass`, `cute_dsl`, and `cutile` are rejected with ROCm migration guidance.

## Documentation

- [Getting Started](docs/GETTING-STARTED.md): prerequisites, installation, first run, and setup issues.
- [Architecture](docs/ARCHITECTURE.md): package layers, data flow, subprocess isolation, and ROCm boundaries.
- [Development](docs/DEVELOPMENT.md): local setup, coding style, source areas, and PR process.
- [Testing](docs/TESTING.md): pytest commands, markers, CI, and hardware-sensitive checks.
- [Configuration](docs/CONFIGURATION.md): CLI flags, benchmark config, environment variables, and Docker settings.
- [Researcher Guide](docs/RESEARCHER-GUIDE.md): workflows for kernel, compiler/backend, agent, and reproducibility researchers.
- [Cookbook](docs/COOKBOOK.md): task-oriented commands for common benchmark workflows.
- [ROCm Notes](docs/rocm.md): host, Docker, and validation notes.
- [ROCm Timing](docs/rocm_timing.md): HIP event timing and optional profiler evidence.
- [ROCm Toolchain Routing](docs/rocm_toolchain_routing.md): evidence tool selection and claim boundaries.
- [Static Kernel Evidence](docs/static_kernel_evidence.md): diagnostic static artifact sidecars.
- [Original Parity](docs/original_parity.md): CUDA-to-ROCm parity boundaries and deferred claims.

Schema-specific references:

- [Definition schema](docs/definition.md)
- [Workload schema](docs/workload.md)
- [Solution schema](docs/solution.md)
- [Trace schema](docs/trace.md)

## Development

Run checks locally with:

```bash
uv run ruff check .
uv run ty check
uv run pytest tests/
```

GPU-sensitive checks use pytest markers such as `requires_rocm`,
`requires_rocm_dev`, `requires_rdna4`, `requires_cdna3`, `requires_ck`,
`requires_rocwmma`, and `timing_serial`.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Contributions should start from an
approved GitHub issue, keep pull requests focused, include tests and
documentation for public behavior changes, and use DCO-signed commits.

## License

Apache License 2.0. See [LICENSE](LICENSE) and [docs/compliance.md](docs/compliance.md).
