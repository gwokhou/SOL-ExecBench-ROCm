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

For auditable ready-subset dataset runs, pass closure and evidence sidecars:

```bash
uv run scripts/run_dataset.py data/SOL-ExecBench/benchmark \
  --ready-subset out/ready_subset.json \
  --readiness out/readiness.json \
  --execution-closure out/execution_closure.json \
  --limit 5
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

If evaluation exits without parseable trace JSONL, the CLI writes a bounded
diagnostic-only no-trace sidecar next to `--output`, in the kept staging
directory, or in the system temp directory. This sidecar records stdout/stderr
tails and line counts; it is not canonical trace JSONL.

### Security Boundary

SOL ExecBench stages and executes submitted solution code in a local
subprocess, with static source review and runtime reward-hack checks before and
during evaluation. These guardrails are not a hardened sandbox and are not a
multi-tenant isolation boundary. Run untrusted submissions only inside an
appropriate container, VM, or isolated ROCm host that you control.

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

- [Provenance Policy](docs/provenance.md): upstream attribution, project-owned ROCm work, SPDX header policy, and non-endorsement boundaries.
- [Compliance](docs/compliance.md): Apache-2.0 license, third-party dependencies, unsupported NVIDIA runtime features, and known gaps.
- [v1.25 Engineering Prerelease Notes](docs/v1_25_release_notes.md): shipped capability, validation evidence, known limitations, and deferred claims.
- [v1.25 Prerelease Checklist](docs/v1_25_prerelease_checklist.md): maintainer checklist from clean tree to tagged release candidate.
- [v1.26 Artifact Bundle](docs/prerelease_artifact_bundle.md): versioned prerelease artifact bundle generation and authority classes.
- [v1.26 Readiness Gates](docs/prerelease_readiness.md): prerelease gate for required artifacts, checksums, claim boundaries, and known gaps.
- [v1.26 Research Preview](docs/research_preview.md): methodology, evidence surfaces, representative commands, and non-claims.
- [v1.26 Public Prerelease Guide](docs/public_prerelease.md): public release-page checklist and publishing wording.
- [Getting Started](docs/GETTING-STARTED.md): prerequisites, installation, first run, and setup issues.
- [Architecture](docs/ARCHITECTURE.md): package layers, data flow, subprocess isolation, and ROCm boundaries.
- [Development](docs/DEVELOPMENT.md): local setup, coding style, source areas, and PR process.
- [Testing](docs/TESTING.md): pytest commands, markers, CI, and hardware-sensitive checks.
- [Configuration](docs/CONFIGURATION.md): CLI flags, benchmark config, environment variables, and Docker settings.
- [Analysis](docs/analysis.md): trace analysis, dataset closure, failure-mode, and sharding semantics.
- [Researcher Guide](docs/RESEARCHER-GUIDE.md): workflows for kernel, compiler/backend, agent, and reproducibility researchers.
- [Cookbook](docs/COOKBOOK.md): task-oriented commands for common benchmark workflows.
- [ROCm Notes](docs/rocm.md): host, Docker, and validation notes.
- [Claims](docs/CLAIMS.md): support, evidence, and forbidden claim boundaries.
- [ROCm Timing](docs/rocm_timing.md): HIP event timing and optional profiler evidence.
- [ROCm Toolchain Routing](docs/rocm_toolchain_routing.md): evidence tool selection and claim boundaries.
- [Static Kernel Evidence](docs/static_kernel_evidence.md): diagnostic static artifact sidecars.
- [Original Parity](docs/original_parity.md): CUDA-to-ROCm parity boundaries and deferred claims.

For first-run troubleshooting, start with [Getting Started](docs/GETTING-STARTED.md)
and then use [Configuration](docs/CONFIGURATION.md) for no-trace diagnostics,
sidecar paths, Docker settings, and environment variables.

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

Focused CPU-safe checks for recent dataset trustworthiness helpers:

```bash
uv run pytest \
  tests/sol_execbench/test_dataset_run_closure.py \
  tests/sol_execbench/test_run_dataset_execution_closure.py \
  tests/sol_execbench/test_dataset_sharding.py
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Contributions should start from an
approved GitHub issue, keep pull requests focused, include tests and
documentation for public behavior changes, and use DCO-signed commits.

## License

Apache License 2.0. See [LICENSE](LICENSE) and [docs/compliance.md](docs/compliance.md).
