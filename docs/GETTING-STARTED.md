<!-- generated-by: gsd-doc-writer -->
# Getting Started

This guide prepares a checkout to run SOL ExecBench ROCm Port against the
included sample problems and, when available, a ROCm-capable AMD GPU.

## Prerequisites

| Requirement | Detail |
| --- | --- |
| Python | `.python-version` uses Python 3.12; `pyproject.toml` allows `>=3.12,<3.14`. |
| Package manager | `uv`. |
| ROCm runtime | Required for live GPU evaluation. The project baseline is ROCm 7.x. |
| GPU access | `/dev/kfd` and `/dev/dri` must be visible for ROCm hardware tests and evaluation. |
| Optional Docker runtime | Native Linux Docker with ROCm device passthrough. |
| Optional dataset download | Hugging Face CLI, available through `huggingface-hub[cli]`. |

On Linux and Windows, dependency resolution uses the PyTorch ROCm 7.1 index for
`torch==2.10.0+rocm7.1` and `torchvision==0.25.0+rocm7.1`. On Linux,
`triton-rocm==3.6.0` resolves from the PyTorch ROCm package root.

## Installation Steps

```bash
git clone https://github.com/gwokhou/SOL-ExecBench-ROCm.git
cd SOL-ExecBench-ROCm
uv sync --all-groups
```

Confirm the package entry points are available:

```bash
uv run sol-execbench contract --json
uv run sol-execbench doctor --json
```

`contract` is GPU-free compatibility metadata. `doctor` probes ROCm tools and
PyTorch ROCm availability and may report missing GPU access in sandboxed
environments.

## Check ROCm Visibility

On a ROCm host, PyTorch should report a HIP build and visible AMD device:

```bash
uv run python -c "import torch; print(torch.__version__); print(torch.version.hip); print(torch.cuda.is_available())"
```

If a device is visible, this also prints the detected architecture:

```bash
uv run python -c "import torch; print(torch.cuda.get_device_properties(0).gcnArchName)"
```

PyTorch ROCm intentionally exposes GPU APIs through the historical
`torch.cuda` namespace.

## First Run

Run an included PyTorch example:

```bash
uv run sol-execbench examples/pytorch/gemma3_swiglu \
  --solution examples/pytorch/gemma3_swiglu/solution_python.json
```

Run an included Triton ROCm example:

```bash
uv run sol-execbench examples/triton/rmsnorm \
  --solution examples/triton/rmsnorm/solution_triton.json
```

Write trace JSONL:

```bash
uv run sol-execbench tests/sol_execbench/samples/linear_backward \
  --solution tests/sol_execbench/samples/linear_backward/solution_python.json \
  --output out/linear_backward.trace.jsonl
```

## Docker Path

Build and enter the default ROCm Docker target:

```bash
./scripts/run_docker.sh --build
```

Run a benchmark command inside the container:

```bash
./scripts/run_docker.sh -- sol-execbench examples/pytorch/gemma3_swiglu \
  --solution examples/pytorch/gemma3_swiglu/solution_python.json
```

Select a declared Docker target:

```bash
./scripts/run_docker.sh --target rocm-7.1.1-ubuntu-24.04-container -- sol-execbench contract --json
```

The wrapper uses `docker/rocm-targets.json` to preview target images and
dependency policies. It also performs runtime and dependency preflight checks
before launching the container.

## Dataset Setup

Download benchmark assets into `data/`:

```bash
uv run --with "huggingface-hub[cli]" ./scripts/download_data.sh
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

Run a bounded dataset batch:

```bash
uv run scripts/run_dataset.py data/SOL-ExecBench/benchmark --limit 5
```

## Common Setup Issues

| Symptom | Check |
| --- | --- |
| `torch.version.hip` is empty | Confirm the ROCm wheel indexes in `pyproject.toml` were used during `uv sync --all-groups`. |
| `torch.cuda.is_available()` is false | Check ROCm installation, GPU visibility, and access to `/dev/kfd` and `/dev/dri`. |
| Hardware tests skip in a sandbox | Run GPU checks on a ROCm host or in Docker with ROCm device passthrough. |
| HIP/C++ examples fail to compile | Confirm ROCm headers and `hipcc` are installed, or use the Docker image. |
| Docker cannot see the GPU | Use native Linux Docker and the repository wrapper instead of Docker Desktop. |
| Dataset download cannot find `hf` | Run through `uv run --with "huggingface-hub[cli]" ./scripts/download_data.sh`. |

## Next Steps

- Read [Architecture](ARCHITECTURE.md) for package layers and subprocess flow.
- Read [Configuration](CONFIGURATION.md) before changing benchmark or Docker settings.
- Read [Development](DEVELOPMENT.md) before editing code.
- Read [Testing](TESTING.md) before running hardware-sensitive checks.
