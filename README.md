<!-- generated-by: gsd-doc-writer -->

# SOL ExecBench ROCm Port

[![Version](https://img.shields.io/badge/version-1.0.2-blue.svg)](pyproject.toml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

A ROCm-only fork of SOL ExecBench for evaluating LLM-generated GPU kernel
solutions on AMD hardware. It supports PyTorch ROCm, Triton ROCm, HIP/C++,
and selected ROCm library categories while rejecting legacy CUDA/NVIDIA
runtime metadata.

This project is independent and is not endorsed by NVIDIA or AMD. See
[Provenance Policy](docs/provenance.md), [Compliance](docs/compliance.md),
and [Claims](docs/CLAIMS.md) for upstream attribution, licensing, validation
boundaries, and hardware-specific claim limits.

## Installation

```bash
uv sync --all-groups
```

Requires Python >=3.12, <3.14 and the `uv` package manager. On Linux x86_64,
dependencies resolve to PyTorch `2.10.0+rocm7.1`, torchvision `0.25.0+rocm7.1`,
and `triton-rocm==3.6.0`. Other platforms use non-ROCm PyTorch wheels so
CPU-safe development tasks still work.

Build and enter the ROCm Docker environment (optional):

```bash
./scripts/run_docker.sh --build
```

## Quick Start

1. Install dependencies:

   ```bash
   uv sync --all-groups
   ```

2. Run an included PyTorch example:

   ```bash
   uv run sol-execbench examples/pytorch/gemma3_swiglu \
     --solution examples/pytorch/gemma3_swiglu/solution_python.json
   ```

3. View the Rich output table showing pass/fail status, latency, speedup, and
   numerical correctness.

## Usage Examples

**Evaluate a PyTorch solution against a problem directory:**

```bash
uv run sol-execbench examples/pytorch/gemma3_swiglu \
  --solution examples/pytorch/gemma3_swiglu/solution_python.json
```

**Evaluate a Triton ROCm solution:**

```bash
uv run sol-execbench examples/triton/rmsnorm \
  --solution examples/triton/rmsnorm/solution_triton.json
```

**Evaluate with explicit definition, workload, and solution files:**

```bash
uv run sol-execbench \
  --definition definition.json \
  --workload workload.jsonl \
  --solution solution.json
```

**Write trace JSONL to a file:**

```bash
uv run sol-execbench examples/pytorch/gemma3_swiglu \
  --solution examples/pytorch/gemma3_swiglu/solution_python.json \
  --output traces.jsonl
```

**Compare two trace JSONL files (baseline comparison):**

```bash
uv run sol-execbench-baseline \
  --candidate traces_v2.jsonl \
  --baseline traces_v1.jsonl
```

**Export a measured HIP baseline registry from trace JSONL:**

```bash
uv run sol-execbench baseline export \
  --trace traces.jsonl \
  --output baseline_registry.json \
  --target-id gemm
```

**Print ROCm environment diagnostics or the evaluator contract:**

```bash
uv run sol-execbench doctor --json
uv run sol-execbench contract --json
uv run sol-execbench toolchain --json
```

**Migrate downloaded SOL-ExecBench or FlashInfer Trace inputs:**

```bash
uv run sol-execbench dataset migrate-sol data/source data/benchmark \
  --manifest out/manifest.json
uv run sol-execbench dataset migrate-flashinfer data/source data/benchmark \
  --manifest out/manifest.json
```

**Run a downloaded dataset batch:**

```bash
uv run scripts/run_dataset.py data/SOL-ExecBench/benchmark --limit 5
```

If evaluation exits without parseable trace JSONL, the CLI writes a bounded
diagnostic-only no-trace sidecar next to `--output`, in the kept staging
directory, or in the system temp directory. This sidecar records stdout/stderr
tails and line counts. It is not canonical trace JSONL.

## CLI Reference

The `sol-execbench` command provides several subcommands:

| Subcommand | Description |
| --- | --- |
| `<problem_dir> --solution <path>` | Evaluate a solution (default when no subcommand is given) |
| `contract --json` | Print the GPU-free evaluator compatibility contract |
| `doctor --json` | Print ROCm environment diagnostics |
| `toolchain --json` | Print ROCm toolchain routing diagnostics |
| `baseline export` | Export a measured HIP baseline registry from SOL trace JSONL |
| `dataset migrate-sol` | Migrate downloaded SOL-ExecBench inputs into local layout |
| `dataset migrate-flashinfer` | Migrate downloaded FlashInfer trace inputs into local layout |

The `sol-execbench-baseline` command compares two trace JSONL files for
baseline regression analysis. It accepts repeated `--baseline` inputs,
`--format text|json`, `--output`, `--win-pct`, `--parity-pct`, and
`--amd-native-claim` for guarded AMD-native reporting.

Key evaluation options:

| Flag | Default | Description |
| --- | --- | --- |
| `--compile-timeout` | 120 | Compilation timeout in seconds (HIP/C++ only) |
| `--timeout` | 600 | Evaluation subprocess timeout in seconds |
| `-o`, `--output` | -- | Write trace JSONL to a file |
| `--json` | off | Print trace JSON lines to stdout |
| `--lock-clocks` | off | Require GPU clocks to be locked for stable timing |
| `--keep-staging` | off | Preserve the temporary staging directory |
| `--profile` | `none` | Collect diagnostic profiling artifacts (`rocprofv3`) |
| `--static-evidence` | `none` | Collect diagnostic static kernel evidence (`auto`) |
| `-v`, `--verbose` | off | Show subprocess output and staging details |

## Supported Solution Categories

The ROCm schema accepts Python/Triton categories and native ROCm categories:

| Category | Notes |
| --- | --- |
| `pytorch` | Python implementation using PyTorch operators |
| `triton` | Triton ROCm implementation |
| `hip_cpp` | Native HIP/C++ implementation built through PyTorch extensions |
| `hipblas` | Native ROCm implementation backed by hipBLAS |
| `miopen` | Native ROCm implementation backed by MIOpen |
| `ck` | Native ROCm implementation using Composable Kernel |
| `rocwmma` | Native ROCm implementation using rocWMMA |

Legacy CUDA/NVIDIA schema values such as `cuda_cpp`, `cublas`, `cudnn`,
`cutlass`, `cute_dsl`, and `cutile` are rejected with ROCm migration guidance.

## Security Boundary

SOL ExecBench stages and runs submitted solution code in a local subprocess.
It uses static source review and runtime reward-hack checks, but these checks
are not a hardened sandbox and are not a multi-tenant isolation boundary. Run
untrusted submissions only inside a container, VM, or isolated ROCm host that
you control.

## Documentation

Start here:

- [Getting Started](docs/GETTING-STARTED.md) -- prerequisites, installation,
  first run, and setup issues
- [Cookbook](docs/COOKBOOK.md) -- task-oriented commands for common benchmark
  workflows
- [Researcher Guide](docs/RESEARCHER-GUIDE.md) -- workflows for kernel,
  compiler/backend, agent, and reproducibility researchers
- [Configuration](docs/CONFIGURATION.md) -- CLI flags, benchmark config,
  environment variables, Docker settings, and no-trace diagnostics
- [Evaluator Contract](docs/EVALUATOR-CONTRACT.md) -- GPU-free compatibility
  contract for evaluator integrations

Project and development references:

- [Architecture](docs/ARCHITECTURE.md) -- package layers, data flow, subprocess
  isolation, and ROCm boundaries
- [Development](docs/DEVELOPMENT.md) -- local setup, coding style, source areas,
  and PR process
- [Testing](docs/TESTING.md) -- pytest commands, markers, CI, and
  hardware-sensitive checks
- [Analysis](docs/analysis.md) -- trace analysis, dataset closure, failure modes,
  and sharding
- [ROCm Notes](docs/rocm.md) -- host, Docker, and validation notes
- [ROCm Timing](docs/rocm_timing.md) -- timing backends, profiler collection,
  and source-specific semantics
- [ROCm Toolchain Routing](docs/rocm_toolchain_routing.md) -- routing rules for
  profiling, static evidence, compiler, and runtime artifacts
- [ROCm Libraries](docs/rocm_libraries.md) -- supported ROCm library categories
  (hipBLAS, MIOpen, Composable Kernel, rocWMMA) for RDNA 4 and CDNA 3 targets;
  CDNA 4-specific low-precision benchmark adaptation is deferred until suitable
  hardware evidence is available. CDNA 4 validation is also deferred.
- [Cookbook](docs/COOKBOOK.md) -- common troubleshooting and task-oriented
  commands for benchmark runs, dataset batches, and evidence workflows

Validation, release, and provenance:

- [Claims](docs/CLAIMS.md) -- support, evidence, and forbidden claim boundaries
- [Provenance Policy](docs/provenance.md) -- upstream attribution, project-owned
  ROCm work, SPDX policy, and non-endorsement boundaries
- [Compliance](docs/compliance.md) -- Apache-2.0 license, dependencies, and
  known gaps
- [Research Preview](docs/research_preview.md) -- methodology, evidence surfaces,
  representative commands, and non-claims
- [v1.25 Engineering Prerelease](docs/v1_25_release_notes.md) -- release notes,
  checklist, and validation evidence
- [Prerelease Artifact Bundle](docs/prerelease_artifact_bundle.md) -- versioned
  artifact bundle and reproducibility closure
- [Prerelease Readiness](docs/prerelease_readiness.md) -- release readiness
  gates and quality checks
- [Public Prerelease](docs/public_prerelease.md) -- public release materials
  and navigation
- [v1.25 Prerelease Checklist](docs/v1_25_prerelease_checklist.md) -- shipping
  checklist and verification steps

Schema-specific references:

- [Definition schema](docs/definition.md)
- [Workload schema](docs/workload.md)
- [Solution schema](docs/solution.md)
- [Trace schema](docs/trace.md)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

Apache License 2.0. See [LICENSE](LICENSE) and
[docs/compliance.md](docs/compliance.md).
