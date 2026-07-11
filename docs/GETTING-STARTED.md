<!-- generated-by: gsd-doc-writer -->
# Getting Started

This guide prepares a checkout to run SOL ExecBench ROCm Port against the
included sample problems and, when available, a ROCm-capable AMD GPU.

## Prerequisites

| Requirement | Detail |
| --- | --- |
| Python | `.python-version` uses Python 3.12; `pyproject.toml` allows `>=3.12,<3.14`. |
| Package manager | `uv`. |
| ROCm runtime | Required for live GPU evaluation. The project baseline is ROCm >= 7.0. |
| GPU access | `/dev/kfd` and `/dev/dri` must be visible for ROCm hardware tests and evaluation. |
| Optional Docker runtime | Native Linux Docker with ROCm device passthrough; the default image is `rocm/dev-ubuntu-24.04:7.1.1-complete`. |
| Optional dataset download | Hugging Face CLI, available through `huggingface-hub[cli]`. |

On Linux x86_64, dependency resolution uses the PyTorch ROCm 7.1 index for
`torch==2.10.0+rocm7.1`, `torchvision==0.25.0+rocm7.1`, and
`triton-rocm==3.6.0`. Non-Linux and non-x86_64 environments use non-ROCm
PyTorch wheels for CPU-safe development tasks such as schema checks,
documentation work, and GPU-free CLI metadata commands.

## Installation Steps

```bash
git clone https://github.com/gwokhou/SOL-ExecBench-ROCm.git
cd SOL-ExecBench-ROCm
uv sync --all-groups
```

Confirm the package entry points are available:

```bash
uv run sol-execbench --format json contract evaluator
uv run sol-execbench --format json environment doctor
```

`contract` is GPU-free compatibility metadata. `doctor` probes ROCm tools,
PyTorch ROCm availability, and device visibility. It may report missing GPU
access on CPU-only development machines, in restricted shells, or in containers
without `/dev/kfd` and `/dev/dri` passthrough.

## Check ROCm Visibility

On a ROCm host, PyTorch should report a HIP build and a visible AMD device:

```bash
uv run python -c "import torch; print(torch.__version__); print(torch.version.hip); print(torch.cuda.is_available())"
```

If a device is visible, this also prints the detected architecture:

```bash
uv run python -c "import torch; print(torch.cuda.get_device_properties(0).gcnArchName)"
```

For the current ROCm port targets, common architecture strings include RDNA 4
`gfx1200` and CDNA3 `gfx942`. PyTorch ROCm intentionally exposes GPU APIs
through the historical `torch.cuda` namespace. In this project that is a
PyTorch ROCm compatibility namespace, not NVIDIA CUDA runtime support.

## First Run

### First-Run Checklist

Use this path for the smallest end-to-end smoke from a fresh checkout on a
ROCm-capable host. The benchmark command in step 4 is the first command that
produces working benchmark output: `out/first-run.trace.jsonl`.

1. Install dependencies with `uv sync --all-groups`.
2. Confirm CLI metadata with `uv run sol-execbench --format json contract evaluator`.
3. Check ROCm visibility with `uv run sol-execbench --format json environment doctor`.
4. Run one included sample and write canonical Trace JSONL:

```bash
mkdir -p out
uv run sol-execbench evaluate tests/sol_execbench/samples/linear_backward \
  --solution tests/sol_execbench/samples/linear_backward/solution_python.json \
  --trace-output out/first-run.trace.jsonl
```

The output file `out/first-run.trace.jsonl` is the canonical benchmark trace
for this first run. Optional environment, profile, static, Matrix, closure,
consistency, claim-upgrade, trust-summary, and release-candidate validation
files are sidecars; they do not replace canonical Trace JSONL. A passing trace
shows that this local command completed on the recorded environment. It is not
full-suite validation.

Run an included PyTorch example:

```bash
uv run sol-execbench evaluate examples/pytorch/gemma3_swiglu \
  --solution examples/pytorch/gemma3_swiglu/solution_python.json
```

Run an included Triton ROCm example:

```bash
uv run sol-execbench evaluate examples/triton/rmsnorm \
  --solution examples/triton/rmsnorm/solution_triton.json
```

Write trace JSONL:

```bash
uv run sol-execbench evaluate tests/sol_execbench/samples/linear_backward \
  --solution tests/sol_execbench/samples/linear_backward/solution_python.json \
  --trace-output out/linear_backward.trace.jsonl
```

### Reading The First Trace

Open `out/first-run.trace.jsonl` and inspect one JSON object per line:

- `evaluation.status` (`status`) reports the evaluation outcome, such as
  `PASSED`, `INCORRECT_NUMERICAL`, `COMPILE_ERROR`, `RUNTIME_ERROR`, or
  `TIMEOUT`.
- `evaluation.correctness` (`correctness`) records numerical comparison fields
  such as `max_relative_error`, `max_absolute_error`, `has_nan`, and `has_inf`.
- `evaluation.performance.latency_ms` (`latency_ms`) is the measured solution
  latency in milliseconds when the trace passes.
- `evaluation.performance.speedup_factor` (`speedup_factor`) is
  `reference_latency_ms / latency_ms` for the same workload and hardware.
- `evaluation.environment` (`environment`) records the AMD/ROCm hardware and
  software context embedded in the canonical Trace JSONL.

If no parseable trace is produced, the CLI writes a bounded no-trace
diagnostics sidecar next to the requested output path, for example
`<output-name>.no-trace-diagnostics.json`. Treat that sidecar as
diagnostic-only evidence and inspect its bounded stdout/stderr tails before
rerunning.

## Docker Path

Build and enter the default ROCm Docker target:

```bash
./scripts/run_docker.sh --build
```

Run a benchmark command inside the container:

```bash
./scripts/run_docker.sh -- sol-execbench evaluate examples/pytorch/gemma3_swiglu \
  --solution examples/pytorch/gemma3_swiglu/solution_python.json
```

Select a declared Docker target:

```bash
./scripts/run_docker.sh --target rocm-7.1.1-ubuntu-24.04-container -- sol-execbench --format json contract evaluator
```

The wrapper uses `docker/rocm-targets.json` to preview target images and
dependency policies. It also performs runtime and dependency preflight checks
before launching the container.

## Dataset Setup

Download benchmark assets into `data/`:

```bash
uv run --with "huggingface-hub[cli]" ./scripts/download_data.sh
```

The wrapper downloads the public SOL-ExecBench dataset into
`data/SOL-ExecBench/benchmark` and FlashInfer Trace files into
`data/flashinfer-trace`. It requires network access and any Hugging Face access
needed by the source repositories. To download only the SOL-ExecBench layout and
write an acquisition manifest, use the Python downloader directly:

```bash
mkdir -p out
uv run python scripts/download_solexecbench.py \
  --output-root data/SOL-ExecBench/benchmark \
  --manifest out/dataset_manifest.json
```

Inspect or verify a local dataset layout without running GPU evaluation:

```bash
uv run python scripts/download_solexecbench.py \
  --verify-only \
  --output-root data/SOL-ExecBench/benchmark \
  --manifest out/dataset_manifest.json
```

This manifest is a sidecar acquisition/layout artifact. It does not prove ROCm
readiness, execution success, paper-level validation, hosted leaderboard parity,
or upstream SOLAR equivalence.

If you already have local source copies, migrate them into the benchmark layout
with the CLI dataset commands:

```bash
uv run sol-execbench dataset migrate sol data/SOL-ExecBench/source data/SOL-ExecBench/benchmark \
  --manifest out/sol-migration-manifest.json
uv run sol-execbench dataset migrate flashinfer data/flashinfer-trace/source data/flashinfer-trace/benchmark \
  --manifest out/flashinfer-migration-manifest.json
```

These commands write generated local artifacts and migration manifests. They do
not grant permission to publish or redistribute NVIDIA SOL-ExecBench source
dataset payloads, FlashInfer Trace source payloads, or derivatives.

Run a bounded dataset batch:

```bash
uv run scripts/run_dataset.py data/SOL-ExecBench/benchmark --limit 5
```

After traces exist, regenerate derived AMD reports without launching the GPU
evaluator:

```bash
uv run scripts/run_dataset.py data/SOL-ExecBench/benchmark \
  --phase derived \
  --jobs auto \
  --amd-score-report out/amd-score-report.json
```

If you are working on RDNA4 profiler-backed timing replacement evidence, build
coverage from timing sidecars:

```bash
uv run python scripts/internal/rdna4/run_rdna4_profiler_timing_coverage.py \
  --timing-evidence-dir out/rdna4-profiler-workload-aggregate-20260608-v2/timing
```

For incomplete profiler-backed targets, use
`scripts/internal/rdna4/run_rdna4_profiler_timing_batch.py --workload-sharded` to profile
workloads independently and aggregate only complete workload manifests. Use
`scripts/internal/rdna4/run_rdna4_profiler_partial_failures.py` and
`scripts/internal/rdna4/run_rdna4_profiler_sharded_closure.py` to classify remaining partial
or blocked targets before making closure claims.

Run a bounded ready-subset batch with an execution-closure report:

```bash
uv run scripts/run_dataset.py data/SOL-ExecBench/benchmark \
  --ready-subset out/ready_subset.json \
  --readiness out/readiness.json \
  --execution-closure out/execution_closure.json \
  --limit 5
```

The closure sidecar records attempted, skipped, filtered, missing-trace, and
missing-evidence states. Existing passing traces are reused only when rerun is
not requested and any prior closure provenance matches the current selected
run. Multi-worker `--jobs` is honored only for `--phase derived`, which reuses
existing trace files and materializes CPU/I/O-only reports; trace collection
and profiler-backed timing remain serial.

## Common Setup Issues

| Symptom | Check |
| --- | --- |
| `uv` is not found | Install `uv` first, then rerun `uv sync --all-groups`. |
| `torch.version.hip` is empty | Confirm the ROCm wheel indexes in `pyproject.toml` were used during `uv sync --all-groups`. |
| `torch.cuda.is_available()` is false | Check ROCm installation, GPU visibility, and access to `/dev/kfd` and `/dev/dri`. |
| Hardware tests skip in a sandbox | Run GPU checks on a ROCm host or in Docker with ROCm device passthrough. |
| HIP/C++ examples fail to compile | Confirm ROCm headers and `hipcc` are installed, or use the Docker image. |
| Docker cannot see the GPU | Use native Linux Docker and the repository wrapper instead of Docker Desktop. |
| Dataset download cannot find `hf` | Run through `uv run --with "huggingface-hub[cli]" ./scripts/download_data.sh`. |
| Evaluation exits with no trace JSONL | Inspect the printed no-trace diagnostics sidecar path; it contains bounded stdout/stderr tails outside canonical trace JSONL. |
| First-run output exists but performance is surprising | Review `evaluation.environment`, clock policy evidence, optional sidecars, and known limitations before treating the result as a performance claim. |

Known limitations for the first run:

- A passing sample trace is ROCm-port evidence for that local command only, not
  full 235-problem paper validation.
- Docker/container ROCm user-space evidence is not native-host validation.
- MI300X and MI308X are sibling GPU products under the CDNA3 architecture
  family and share the `gfx942` code path. Current CDNA3/gfx942 validation
  infrastructure evidence was recorded on MI308X, not MI300X, so full-suite
  MI300X validation remains blocked until timeout, clock-lock, timing, score,
  FP8, low-precision, and exact-hardware evidence boundaries are resolved.
  CDNA4 validation is unavailable because suitable hardware is not currently
  accessible.
- NVFP4/MXFP4 Quant ROCm adaptation and hardware validation are deferred until
  CDNA4-class hardware is available. CDNA3 runs should treat those workloads as
  documented hardware-unsupported skips, not CPU, dequantized, or paper-level
  benchmark validation.
- PyTorch and some internal APIs may still contain CUDA-named compatibility
  symbols; those names do not mean this ROCm-only port supports NVIDIA CUDA
  runtime execution.

## Next Steps

- Read [Architecture](ARCHITECTURE.md) for package layers and subprocess flow.
- Read [Configuration](CONFIGURATION.md) before changing benchmark or Docker settings.
- Read [Development](DEVELOPMENT.md) before editing code.
- Read [Testing](TESTING.md) before running hardware-sensitive checks.
