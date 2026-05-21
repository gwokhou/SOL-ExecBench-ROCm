# SOL ExecBench ROCm Port

SOL ExecBench ROCm Port evaluates AI-generated GPU kernel solutions on AMD
ROCm hardware while preserving the original SOL ExecBench problem, workload,
solution, and trace semantics where practical.

This repository is ROCm-only. It does not maintain CUDA/NVIDIA runtime support.
Legacy NVIDIA-specific solution categories are rejected by the ROCm schema or
kept only as documented fallback examples.

[HuggingFace Dataset](https://huggingface.co/datasets/nvidia/SOL-ExecBench) |
[Original Leaderboard](https://research.nvidia.com/benchmarks/sol-execbench) |
[Technical Report](https://arxiv.org/abs/2603.19173)

Kernels are:

- checked for reward hacking,
- tested against a PyTorch reference solution for numerical correctness,
- timed under reproducible ROCm-compatible conditions.

Supported solution categories in this port:

- PyTorch on ROCm
- Triton ROCm
- HIP/C++
- ROCm library-oriented native categories: `hipblas`, `miopen`, `ck`, `rocwmma`

Current local validation evidence covers RDNA 4 (`gfx1200`). CDNA 3 validation
is a deferred hardware follow-up and should not be claimed until a full
adapted-suite run is recorded on `gfx94*`.

## Prerequisites

- Linux host with ROCm 7.0 or newer and a visible AMD GPU.
- Native Linux Docker daemon for container use. Docker Desktop cannot pass
  `/dev/kfd` and `/dev/dri` through correctly for ROCm evaluation.
- Access to `/dev/kfd` and `/dev/dri` on the host.
- `uv` for local Python environment management.
- Hugging Face CLI for benchmark dataset downloads:

```bash
pip install "huggingface-hub[cli]"
```

See [ROCm setup](docs/rocm.md) for detailed host, Docker, and validation notes.

## Setup

Install the Python environment:

```bash
uv sync --all-groups
```

Verify that PyTorch is using ROCm wheels:

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

`torch.version.hip` should be set and `torch.version.cuda` should be `None`.

Download benchmark data:

```bash
./scripts/download_data.sh
```

This downloads the original SOL-ExecBench dataset and FlashInfer trace dataset
into `data/`.

## Docker

Build and enter the ROCm container:

```bash
./scripts/run_docker.sh --build
```

Run a command inside the container:

```bash
./scripts/run_docker.sh -- sol-execbench tests/sol_execbench/samples/rmsnorm \
  --solution tests/sol_execbench/samples/rmsnorm/solution_python.json
```

The script mounts the repository at `/sol-execbench`, passes `/dev/kfd` and
`/dev/dri`, and sets the FlashInfer trace path when data is present.

## Evaluating A Solution

Use the `sol-execbench` CLI:

```bash
uv run sol-execbench <problem_dir> --solution solution.json
```

or specify files explicitly:

```bash
uv run sol-execbench \
  --definition def.json \
  --workload workload.jsonl \
  --solution solution.json
```

Common options:

| Flag | Description |
| --- | --- |
| `--compile-timeout` | HIP/C++ compilation timeout in seconds. |
| `--timeout` | Evaluation subprocess timeout in seconds. |
| `-o, --output` | Write trace JSONL to a file. |
| `--json` | Print trace JSON to stdout. |
| `--lock-clocks` | Require ROCm GPU clocks to be locked before benchmarking. |
| `--keep-staging` | Preserve the staging directory after a run. |
| `-v, --verbose` | Show subprocess output. |

## Running A Dataset

Evaluate a dataset batch with:

```bash
uv run scripts/run_dataset.py data/SOL-ExecBench/benchmark --limit 5
```

Use a named solution file for each problem:

```bash
uv run scripts/run_dataset.py data/SOL-ExecBench/benchmark \
  --solution-name solution.json
```

Results are written under `out/run_dataset/` by default. Existing passed
problems are skipped unless `--rerun` is supplied.

## Schemas And Analysis

- [Definition](docs/definition.md): kernel specification and PyTorch reference.
- [Workload](docs/workload.md): concrete input configurations and tolerances.
- [Solution](docs/solution.md): ROCm-supported implementation metadata.
- [Trace](docs/trace.md): evaluation output and ROCm environment fields.
- [Analysis](docs/analysis.md): timing, trace review, clock locking, and
  `rocprofv3` workflow.
- [Compliance](docs/compliance.md): third-party notices and unsupported features.

## Testing

Run the full adapted test suite:

```bash
uv run pytest tests/
```

Focused checks:

```bash
uv run pytest tests/sol_execbench/core/data/test_solution.py
uv run pytest tests/examples/test_examples.py -k consistency
uv run ruff check .
```

## Citation

```bibtex
@misc{lin2026solexecbench,
      title={SOL-ExecBench: Speed-of-Light Benchmarking for Real-World GPU Kernels Against Hardware Limits},
      author={Edward Lin, Sahil Modi, Siva Kumar Sastry Hari, Qijing Huang, Zhifan Ye, Nestor Qin, Fengzhe Zhou, Yuan Zhang, Jingquan Wang, Sana Damani, Dheeraj Peri, Ouye Xie, Aditya Kane, Moshe Maor, Michael Behar, Triston Cao, Rishabh Mehta, Vartika Singh, Vikram Sharma Mailthody, Terry Chen, Zihao Ye, Hanfeng Chen, Tianqi Chen, Vinod Grover, Wei Chen, Wei Liu, Eric Chung, Luis Ceze, Roger Bringmann, Cyril Zeller, Michael Lightstone, Christos Kozyrakis, Humphrey Shi},
      year={2026},
      eprint={2603.19173},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2603.19173},
}
```

## License

Apache-2.0. See [LICENSE](LICENSE) and [Compliance](docs/compliance.md).
