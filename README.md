<!-- generated-by: gsd-doc-writer -->

# SOL ExecBench for ROCm

This repository ports SOL ExecBench to the AMD ROCm stack. It is meant for
researchers and low-level GPU developers who want to evaluate LLM-generated
kernels on AMD hardware while staying close to the original SOL ExecBench
benchmark shape.

Beyond the required CUDA-to-ROCm adaptation, this port adds several research and
engineering features that make AMD-side evaluation easier to audit:

- ROCm-native solution categories for HIP/C++, Triton ROCm, hipBLAS, MIOpen,
  Composable Kernel, and rocWMMA.
- AMD-oriented evidence collection for runtime environment, toolchain routing,
  `rocprofv3`, static kernel artifacts, and ROCm compatibility checks.
- Dataset readiness and execution accounting, including ready subsets, closure
  reports, parity gaps, consistency checks, stability summaries, and trust
  summaries.
- AMD score and bound helpers that stay separate from canonical Trace JSONL, so
  benchmark traces and derived ROCm analysis remain distinct.
- Local migration workflows for SOL-ExecBench and FlashInfer Trace inputs, with
  manifest and provenance metadata for auditable dataset handling.
- Release and claim guardrails for checksums, required artifacts, known gaps,
  unsupported cases, provenance classes, and forbidden public claims.

The goal is to make ROCm benchmark runs easier to reproduce, inspect, and
compare without presenting the port as a new leaderboard authority or a
paper-level parity result.

This project is independent and is not endorsed by NVIDIA or AMD. See
[Provenance Policy](docs/provenance.md), [Compliance](docs/compliance.md), and
[Claims](docs/CLAIMS.md) for upstream attribution, licensing, validation
boundaries, and hardware-specific claim limits.

## Requirements

- Python `>=3.12,<3.14`
- `uv` for dependency and environment management
- Linux with ROCm-capable AMD hardware for GPU evaluation
- ROCm device access through `/dev/kfd` and `/dev/dri`
- ROCm 7.x user-space tooling for native HIP/C++ and profiler paths
- Docker, when using the provided ROCm container workflow

On Linux x86_64, dependencies resolve to PyTorch `2.10.0+rocm7.1`,
torchvision `0.25.0+rocm7.1`, and `triton-rocm==3.6.0`. Other platforms use
non-ROCm PyTorch wheels so CPU-safe development tasks still work. The Docker
target manifest covers ROCm 7.0.2, 7.1.1, and 7.2.0 containers for compatibility
and evidence workflows.

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

Run one included PyTorch problem:

```bash
uv run sol-execbench examples/pytorch/gemma3_swiglu \
  --solution examples/pytorch/gemma3_swiglu/solution_python.json
```

Run one Triton ROCm problem:

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

Rebuild derived reports from existing traces without rerunning GPU evaluation:

```bash
uv run scripts/run_dataset.py data/SOL-ExecBench/benchmark \
  --phase derived \
  --jobs auto \
  --amd-score-report out/amd-score-report.json
```

Migrate locally downloaded SOL-ExecBench or FlashInfer Trace inputs into the
repository benchmark layout:

```bash
uv run sol-execbench dataset migrate-sol data/SOL-ExecBench/source data/SOL-ExecBench/benchmark \
  --manifest out/sol-migration-manifest.json
uv run sol-execbench dataset migrate-flashinfer data/flashinfer-trace/source data/flashinfer-trace/benchmark \
  --manifest out/flashinfer-migration-manifest.json
```

These commands organize local inputs. They do not redistribute restricted source
data.

For auditable ready-subset dataset runs, include closure and evidence sidecars:

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
tails and line counts. It is not canonical trace JSONL.

### Security Boundary

SOL ExecBench stages and runs submitted solution code in a local subprocess. It
uses static source review and runtime reward-hack checks, but these checks are
not a hardened sandbox and are not a multi-tenant isolation boundary. Run
untrusted submissions only inside a container, VM, or isolated ROCm host that
you control.

Metadata and diagnostic subcommands:

```bash
uv run sol-execbench contract --json
uv run sol-execbench doctor --json
uv run sol-execbench toolchain --json
uv run sol-execbench toolchain --json --list-registry
uv run sol-execbench dataset migrate-sol --help
uv run sol-execbench dataset migrate-flashinfer --help
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

See [ROCm library examples](docs/rocm_libraries.md) for library readiness,
example coverage, and diagnostic boundaries.

Validation status:

- RDNA4 `gfx1200` bounded ready-subset evidence exists from v1.30 artifacts:
  121 ready problems, 1907 attempted workloads, 1761 passed workloads, 146
  failed workloads, 86 OK problems, 35 FAIL problems, and 12 explicit
  `missing_trace` workload records.
- The RDNA4 derived evidence set contains 1895 score records, with 172 scored
  and 1723 unscored. AMD SOL v2 and SOLAR derivation sidecars cover 1839
  traces after 56 temporary long-tail sidecar exclusions, with zero unexcluded
  sidecars missing.
- RDNA4 timing remains non-authoritative because clock-lock/reset sudoers
  coverage is incomplete and timing sidecars used PyTorch/device-event fallback
  rather than profiler-backed `rocprofv3` kernel activity timing.
- Long RDNA4 derived or dataset jobs should be launched through
  `scripts/run_derived_isolated.py --launch-mode systemd` or an equivalent
  transient `systemd-run --user` unit with `MemoryMax` and `MemorySwapMax`
  caps, then polled by status/log files. This prevents OOM-heavy workloads from
  taking down the calling Codex/session process.
- CDNA3 MI308X (`gfx942`) infrastructure evidence exists for adapted-suite and
  dataset paths, with documented blockers.
- This is not full MI300X hardware validation.
- Full MI300X validation remains blocked until timeout, clock-lock, timing,
  score, FP8, low-precision, and exact-hardware evidence boundaries are
  resolved.
- NVFP4/MXFP4 Quant ROCm adaptation and hardware validation are deferred until
  CDNA4-class hardware is available.
- CDNA3 runs should treat those problems as documented hardware-unsupported
  skips, not CPU or dequantized benchmark validation.
- CDNA4 validation is deferred because suitable hardware is unavailable.

Legacy CUDA/NVIDIA schema values such as `cuda_cpp`, `cublas`, `cudnn`,
`cutlass`, `cute_dsl`, and `cutile` are rejected with ROCm migration guidance.

## Version Names

The repository uses a few different version labels:

- The Python package version comes from `pyproject.toml`.
- Milestone labels such as `v1.25` or `v1.26` describe project planning and
  release-readiness work.
- Prerelease tags such as `v1.26.0-rc1` describe a public review package.

When reading release docs, treat milestone and prerelease labels as evidence
context. They do not change the benchmark schema or upgrade validation claims
unless the cited artifacts and `docs/CLAIMS.md` say so.

## Documentation

Start here:

- [Getting Started](docs/GETTING-STARTED.md): prerequisites, installation, first
  run, and setup issues.
- [Cookbook](docs/COOKBOOK.md): task-oriented commands for common benchmark
  workflows.
- [Researcher Guide](docs/RESEARCHER-GUIDE.md): workflows for kernel,
  compiler/backend, agent, and reproducibility researchers.
- [Configuration](docs/CONFIGURATION.md): CLI flags, benchmark config,
  environment variables, Docker settings, and no-trace diagnostics.

Project and development references:

- [Architecture](docs/ARCHITECTURE.md): package layers, data flow, subprocess
  isolation, and ROCm boundaries.
- [Development](docs/DEVELOPMENT.md): local setup, coding style, source areas,
  and PR process.
- [Testing](docs/TESTING.md): pytest commands, markers, CI, and
  hardware-sensitive checks.
- [Analysis](docs/analysis.md): trace analysis, dataset closure, failure modes,
  and sharding.
- [ROCm Notes](docs/rocm.md): host, Docker, and validation notes.

Validation, release, and provenance:

- [Claims](docs/CLAIMS.md): support, evidence, and forbidden claim boundaries.
- [Provenance Policy](docs/provenance.md): upstream attribution, project-owned
  ROCm work, SPDX policy, history cleanup, and non-endorsement boundaries.
- [Compliance](docs/compliance.md): Apache-2.0 license, dependencies,
  unsupported NVIDIA runtime features, and known gaps.
- [Research Preview](docs/research_preview.md): methodology, evidence surfaces,
  representative commands, and non-claims.
- [v1.25 Engineering Prerelease Notes](docs/v1_25_release_notes.md): shipped
  capability, validation evidence, known limitations, and deferred claims.
- [v1.25 Prerelease Checklist](docs/v1_25_prerelease_checklist.md): maintainer
  checklist from clean tree to tagged release candidate.
- [Prerelease Artifact Bundle](docs/prerelease_artifact_bundle.md): prerelease
  artifact bundle generation and authority classes.
- [Prerelease Readiness Gates](docs/prerelease_readiness.md): required
  artifacts, checksums, claim boundaries, provenance, and known gaps.
- [Public Prerelease Guide](docs/public_prerelease.md): release-page checklist
  and publishing wording.

ROCm evidence details:

- [ROCm Timing](docs/rocm_timing.md): HIP event timing and optional profiler
  evidence.
- [ROCm Toolchain Routing](docs/rocm_toolchain_routing.md): evidence tool
  selection and claim boundaries.
- [Static Kernel Evidence](docs/static_kernel_evidence.md): diagnostic static
  artifact sidecars.
- [Original Parity](docs/original_parity.md): CUDA-to-ROCm parity boundaries
  and deferred claims.
- [Dataset provenance and local migration](docs/COOKBOOK.md): local-only
  SOL-ExecBench and FlashInfer Trace migration workflows.

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
`requires_rocm_dev`, `requires_rdna4` for RDNA 4, `requires_cdna3` for CDNA 3,
`requires_ck`, `requires_rocwmma`, and `timing_serial`.
CDNA 4 validation is also deferred until suitable hardware evidence exists.

Focused CPU-safe checks for provenance and prerelease guardrails:

```bash
uv run pytest \
  tests/sol_execbench/test_provenance_policy.py \
  tests/sol_execbench/test_prerelease_readiness.py \
  tests/sol_execbench/test_public_prerelease_docs.py \
  tests/sol_execbench/test_research_preview_docs.py -q
```

Focused CPU-safe checks for recent dataset trustworthiness helpers:

```bash
uv run pytest \
  tests/sol_execbench/test_dataset_run_closure.py \
  tests/sol_execbench/test_run_dataset_execution_closure.py \
  tests/sol_execbench/test_run_dataset_amd_score.py \
  tests/sol_execbench/test_dataset_sharding.py
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Contributions should start from an
approved GitHub issue, keep pull requests focused, include tests and
documentation for public behavior changes, and use DCO-signed commits.

## License

Apache License 2.0. See [LICENSE](LICENSE) and [docs/compliance.md](docs/compliance.md).
