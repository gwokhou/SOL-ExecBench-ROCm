<!-- generated-by: gsd-doc-writer -->
# Getting Started

This guide gets a local ROCm-capable checkout ready to run SOL ExecBench ROCm
Port against a small benchmark problem.

## Prerequisites

| Requirement | Version or Detail |
| --- | --- |
| Python | `3.12` from `.python-version`; `pyproject.toml` allows `>=3.12,<3.14` |
| ROCm | ROCm 7.0 or newer for the project baseline; the Docker image uses `rocm/dev-ubuntu-24.04:7.1.1-complete` |
| GPU | AMD GPU visible to PyTorch ROCm |
| Package manager | `uv` |
| Optional container runtime | Native Linux Docker daemon with `/dev/kfd` and `/dev/dri` access |
| Optional dataset download | Hugging Face CLI command `hf` from `huggingface-hub[cli]` |

## Installation Steps

1. Clone the repository.

```bash
git clone https://github.com/gwokhou/SOL-ExecBench-ROCm.git
cd SOL-ExecBench-ROCm
```

2. Install runtime and development dependencies.

```bash
uv sync --all-groups
```

3. Confirm that PyTorch sees ROCm.

```bash
uv run python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.hip)
print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.get_device_properties(0).gcnArchName)
PY
```

`torch.version.hip` should be set, and `torch.version.cuda` should be `None`.

## First Run

Run a small included problem with the CLI:

```bash
uv run sol-execbench examples/pytorch/gemma3_swiglu \
  --solution examples/pytorch/gemma3_swiglu/solution_python.json
```

The command loads `definition.json`, `workload.jsonl`, and the supplied solution
file from the example directory, evaluates each workload, and prints a Rich
summary table.

## Optional Dataset Setup

Download the benchmark assets when you need dataset-scale runs:

```bash
uv run --with "huggingface-hub[cli]" ./scripts/download_data.sh
```

The SOL-ExecBench downloader writes the public benchmark layout to
`data/SOL-ExecBench/benchmark` by default. To verify or record the local layout
without running GPU evaluation, emit an acquisition/layout manifest:

```bash
uv run python scripts/download_solexecbench.py \
  --verify-only \
  --output-root data/SOL-ExecBench/benchmark \
  --manifest out/dataset_manifest.json
```

This manifest is a sidecar acquisition/layout artifact. It does not prove ROCm readiness,
execution success, paper-level validation, hosted leaderboard parity, or
upstream SOLAR equivalence.

Then run a small dataset batch:

```bash
uv run scripts/run_dataset.py data/SOL-ExecBench/benchmark --limit 5
```

## Optional Docker Setup

Build and enter the ROCm container:

```bash
./scripts/run_docker.sh --build
```

Run a command inside the container:

```bash
./scripts/run_docker.sh -- sol-execbench examples/pytorch/gemma3_swiglu \
  --solution examples/pytorch/gemma3_swiglu/solution_python.json
```

## Common Setup Issues

| Issue | Resolution |
| --- | --- |
| PyTorch reports `torch.version.cuda` instead of `torch.version.hip`. | Re-run `uv sync --all-groups` and confirm the ROCm wheel indexes in `pyproject.toml` are reachable. |
| `torch.cuda.is_available()` returns `False` on a ROCm host. | Confirm ROCm is installed, the current user can access `/dev/kfd` and `/dev/dri`, and the GPU is visible through ROCm tools. |
| Docker cannot see the AMD GPU. | Use a native Linux Docker daemon and `./scripts/run_docker.sh`; Docker Desktop cannot correctly pass ROCm devices through. |
| `./scripts/download_data.sh` cannot find `hf`. | Run the downloader through `uv run --with "huggingface-hub[cli]" ./scripts/download_data.sh` so the Hugging Face CLI is available on `PATH`. |
| HIP/C++ examples fail to compile. | Confirm ROCm compiler tools such as `hipcc` are available, or use the provided Docker image. |

## Next Steps

- Read [Architecture](ARCHITECTURE.md) for the package layers and data flow.
- Read [Configuration](CONFIGURATION.md) for CLI and benchmark settings.
- Read [Development](DEVELOPMENT.md) before changing source code.
- Read [Testing](TESTING.md) before running or adding tests.
