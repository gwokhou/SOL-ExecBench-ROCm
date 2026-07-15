<!-- generated-by: gsd-doc-writer -->

# SOL ExecBench ROCm Port

[![Version](https://img.shields.io/badge/version-2.0.0-blue.svg)](pyproject.toml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

A ROCm-only fork of SOL ExecBench for evaluating LLM-generated GPU kernel
solutions on AMD hardware. It supports PyTorch ROCm, Triton ROCm, HIP/C++,
and selected ROCm library categories while rejecting legacy CUDA/NVIDIA
runtime metadata.

This project is independent and is not endorsed by NVIDIA or AMD. See
[Provenance Policy](docs/user/provenance.md), [Compliance](docs/user/compliance.md),
and [Claims](docs/user/CLAIMS.md) for upstream attribution, licensing, validation
boundaries, and hardware-specific claim limits.

## Installation

```bash
uv sync --all-groups
```

Requires Python >=3.12, <3.14 and the `uv` package manager. On Linux x86_64,
dependencies resolve to PyTorch `2.11.0+rocm7.2`, torchvision `0.26.0+rocm7.2`,
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
   uv run sol-execbench evaluate examples/pytorch/gemma3_swiglu \
     --solution examples/pytorch/gemma3_swiglu/solution_python.json
   ```

3. View the Rich output table showing pass/fail status, latency, speedup, and
   numerical correctness.

## Usage Examples

**Evaluate a PyTorch solution against a problem directory:**

```bash
uv run sol-execbench evaluate examples/pytorch/gemma3_swiglu \
  --solution examples/pytorch/gemma3_swiglu/solution_python.json
```

**Evaluate a Triton ROCm solution:**

```bash
uv run sol-execbench evaluate examples/triton/rmsnorm \
  --solution examples/triton/rmsnorm/solution_triton.json
```

**Evaluate with explicit definition, workload, and solution files:**

```bash
uv run sol-execbench evaluate \
  --definition definition.json \
  --workload workload.jsonl \
  --solution solution.json
```

**Write trace JSONL to a file:**

```bash
uv run sol-execbench evaluate examples/pytorch/gemma3_swiglu \
  --solution examples/pytorch/gemma3_swiglu/solution_python.json \
  --trace-output traces.jsonl
```

**Compare two trace JSONL files (baseline comparison):**

```bash
uv run sol-execbench baseline compare \
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
uv run sol-execbench --format json environment doctor
uv run sol-execbench --format json contract evaluator
uv run sol-execbench --format json toolchain route
```

**Migrate downloaded SOL-ExecBench or FlashInfer Trace inputs:**

```bash
uv run sol-execbench dataset migrate sol data/source data/benchmark \
  --manifest out/manifest.json
uv run sol-execbench dataset migrate flashinfer data/source data/benchmark \
  --manifest out/manifest.json
```

**Run a downloaded dataset batch:**

```bash
uv run scripts/run_dataset.py data/SOL-ExecBench/benchmark --limit 5
```

**Write an AMD-native report for a derived batch:**

```bash
uv run scripts/run_dataset.py data/SOL-ExecBench/benchmark --phase derived \
  --amd-score-report out/amd-score.json \
  --scoring-baseline baselines/gfx1200.json
```

For an official score, use `sol-execbench score official` with the matching
release bundle, independent rerun verification, and suite manifest. Dataset
sidecars requested with `--official-score-report out/official-score.json`
without those release artifacts remain explicitly blocked; use
`--official-aggregation-policy fixed_suite_denominator_zero_for_blocked` for
the only accepted aggregation policy.

If evaluation exits without parseable trace JSONL, the CLI writes a bounded
diagnostic-only no-trace sidecar next to `--trace-output`, in the kept staging
directory, or in the system temp directory. This sidecar records stdout/stderr
tails and line counts. It is not canonical trace JSONL.

## CLI Reference

The `sol-execbench` command provides several subcommands:

| Subcommand | Description |
| --- | --- |
| `evaluate <problem_dir> --solution <path>` | Evaluate a solution |
| `--format json contract evaluator` | Print the GPU-free evaluator compatibility contract |
| `--format json contract cli` | Print the generated CLI contract |
| `--format json environment doctor` | Print ROCm environment diagnostics |
| `--format json toolchain route` | Print ROCm toolchain routing diagnostics |
| `baseline export` | Export a measured HIP baseline registry from SOL trace JSONL |
| `dataset migrate sol` | Migrate downloaded SOL-ExecBench inputs into local layout |
| `dataset migrate flashinfer` | Migrate downloaded FlashInfer trace inputs into local layout |

The `sol-execbench baseline compare` command compares trace JSONL files for
baseline regression analysis. It accepts repeated `--baseline` inputs,
root-level `--format text|json`, `--output`, `--win-pct`, `--parity-pct`, and
`--amd-native-claim` for guarded AMD-native reporting.

Key evaluation options:

| Flag | Default | Description |
| --- | --- | --- |
| `--compile-timeout` | 120 | Compilation timeout in seconds (HIP/C++ only) |
| `--timeout` | 600 | Evaluation subprocess timeout in seconds |
| `--trace-output` | -- | Write canonical trace JSONL to a file |
| `--lock-clocks` | off | Require GPU clocks to be locked for stable timing |
| `--keep-staging` | off | Preserve the temporary staging directory |
| `--profile` | `none` | Collect diagnostic profiling artifacts (`rocprofv3`) |
| `--static-evidence` | `none` | Collect diagnostic static kernel evidence (`auto`) |
| `-v`, `--verbose` | off | Show subprocess output and staging details |

CLI 2.0 is intentionally breaking and does not install compatibility aliases:

| CLI 1.x | CLI 2.0 |
| --- | --- |
| `sol-execbench DIR ...` | `sol-execbench evaluate DIR ...` |
| `sol-execbench doctor --json` | `sol-execbench --format json environment doctor` |
| `sol-execbench contract --json` | `sol-execbench --format json contract evaluator` |
| `sol-execbench-baseline ...` | `sol-execbench baseline compare ...` |
| `baseline release-build` | `baseline release build` |

Root `--format` must precede the subcommand. JSON evaluation also requires
`--trace-output`; stdout carries one response envelope while canonical Trace
JSONL remains a separate artifact.

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

Use the [Documentation Map](docs/README.md) to select user guides, internal
maintainer records, examples, or release evidence.

Start here:

- [Getting Started](docs/user/GETTING-STARTED.md) -- prerequisites, installation,
  first run, and setup issues
- [Cookbook](docs/user/COOKBOOK.md) -- task-oriented commands and troubleshooting
  for common benchmark workflows
- [Researcher Guide](docs/user/RESEARCHER-GUIDE.md) -- workflows for kernel,
  compiler/backend, agent, and reproducibility researchers
- [Configuration](docs/user/CONFIGURATION.md) -- CLI flags, benchmark config,
  environment variables, Docker settings, and no-trace diagnostics
- [Evaluator Contract](docs/user/EVALUATOR-CONTRACT.md) -- GPU-free compatibility
  contract for evaluator integrations

User references:

- [ROCm Notes](docs/user/rocm.md) -- host, Docker, and validation notes
- [ROCm Timing](docs/user/rocm_timing.md) -- timing backends, profiler collection,
  and source-specific semantics
- [ROCm Toolchain Routing](docs/user/rocm_toolchain_routing.md) -- routing rules for
  profiling, static evidence, compiler, and runtime artifacts
- [ROCm Libraries](docs/user/rocm_libraries.md) -- supported ROCm library categories
  (hipBLAS, MIOpen, Composable Kernel, rocWMMA) for RDNA 4 and CDNA 3 targets;
  CDNA 4-specific low-precision benchmark adaptation is deferred until suitable
  hardware evidence is available. CDNA 4 validation is also deferred.

Public policy and evidence references:

- [Claims](docs/user/CLAIMS.md) -- support, evidence, and forbidden claim boundaries
- [Provenance Policy](docs/user/provenance.md) -- upstream attribution, project-owned
  ROCm work, SPDX policy, and non-endorsement boundaries
- [Compliance](docs/user/compliance.md) -- Apache-2.0 license, dependencies, and
  known gaps
- [Research Preview](docs/user/research_preview.md) -- methodology, evidence surfaces,
  representative commands, and non-claims
- [Public Prerelease](docs/user/public_prerelease.md) -- public release materials
  and navigation

Release and maintainer records:

- [v1.25 Engineering Prerelease](docs/internal/v1_25_release_notes.md) and its
  [release checklist](docs/internal/v1_25_prerelease_checklist.md)
- [Prerelease artifact bundle](docs/internal/prerelease_artifact_bundle.md) and
  [prerelease readiness gates](docs/internal/prerelease_readiness.md)
- [Release evidence manifests](docs/releases/) -- version-scoped, non-authoritative
  evidence records

Schema-specific references:

- [Definition schema](docs/user/definition.md)
- [Workload schema](docs/user/workload.md)
- [Solution schema](docs/user/solution.md)
- [Trace schema](docs/user/trace.md)

[Architecture](docs/user/ARCHITECTURE.md), [Development](docs/user/DEVELOPMENT.md),
and [Testing](docs/user/TESTING.md) are public contributor documentation.
Maintainer and historical material, such as [Analysis](docs/internal/analysis.md),
lives in [`docs/internal/`](docs/internal/). Versioned release evidence remains
in [`docs/releases/`](docs/releases/).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

Apache License 2.0. See [LICENSE](LICENSE) and
[docs/user/compliance.md](docs/user/compliance.md).
