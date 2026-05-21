# Analysis And Profiling

SOL ExecBench emits one JSON trace per workload. Use these traces as the
primary analysis artifact, then use ROCm profiling tools for deeper kernel
inspection when needed.

## Trace Collection

Write JSONL traces from a single problem:

```bash
uv run sol-execbench tests/sol_execbench/samples/rmsnorm \
  --solution tests/sol_execbench/samples/rmsnorm/solution_python.json \
  --json \
  -o out/rmsnorm.jsonl
```

Run a dataset batch:

```bash
uv run scripts/run_dataset.py data/SOL-ExecBench/benchmark --limit 5
```

Dataset runs write per-problem traces and a summary under `out/run_dataset/`
unless `-o` is supplied.

## Timing Method

The ROCm port does not use CUPTI. Timing uses PyTorch's HIP-backed device event
API through the historical `torch.cuda.Event` namespace. This is intentional:
PyTorch exposes ROCm GPU devices through `torch.cuda` compatibility APIs.

The benchmark path:

1. pre-allocates distinct input/output buffers for timed iterations,
2. clears a GPU cache-sized tensor before each iteration,
3. records HIP-backed device events around the solution call,
4. synchronizes before reading elapsed time,
5. emits median latency by default.

## Clock Stability

For benchmark-grade runs, use:

```bash
uv run sol-execbench <problem_dir> --solution solution.json --lock-clocks
```

Clock locking uses `rocm-smi`. The command fails the workload if
`--lock-clocks` is requested but the environment did not lock clocks before the
evaluation subprocess starts. This prevents silently mixing locked and unlocked
timing data.

## External ROCm Profiling

Use `rocprofv3` around a small, representative command when kernel-level
analysis is needed:

```bash
rocprofv3 --stats -- \
  uv run sol-execbench tests/sol_execbench/samples/rmsnorm \
    --solution tests/sol_execbench/samples/rmsnorm/solution_python.json
```

For HIP/C++ solutions, keep compilation outside the profiling window when
possible by running once with `--keep-staging`, then profiling a repeated run.
This keeps compile overhead out of kernel timing analysis.

## Interpreting Results

Key trace fields:

- `evaluation.status`: final outcome for the workload.
- `evaluation.correctness`: maximum absolute/relative error and non-finite
  flags.
- `evaluation.performance.latency_ms`: measured solution latency.
- `evaluation.performance.reference_latency_ms`: PyTorch reference latency on
  the same hardware.
- `evaluation.environment`: AMD hardware and ROCm/PyTorch library versions.

Do not compare latencies across machines unless ROCm version, GPU architecture,
clock policy, and problem inputs are comparable.
