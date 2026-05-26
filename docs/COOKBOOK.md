# Cookbook

These recipes are copy-pasteable starting points. Commands that need AMD GPU
access require a ROCm-capable Linux host or the ROCm Docker wrapper.

## Recipe: Single-Kernel Evaluation

Use this recipe for single-kernel evaluation before interpreting wider
curated-slice or dataset results.

```bash
uv run sol-execbench tests/sol_execbench/samples/rmsnorm \
  --solution tests/sol_execbench/samples/rmsnorm/solution_triton.json \
  --json \
  -o out/cookbook/rmsnorm.trace.jsonl
```

Inspect:

```bash
head -n 1 out/cookbook/rmsnorm.trace.jsonl
```

## Recipe: Add A HIP/C++ Solution

This recipe adds a HIP/C++ solution while keeping the benchmark problem stable.

1. Copy the nearest example, such as `examples/hip_cpp/rmsnorm/`.
2. Keep `<problem_dir>/definition.json` and `<problem_dir>/workload.jsonl` stable for comparison.
3. Put native source in the solution JSON `sources` block.
4. Use `languages: ["hip_cpp"]` and AMD target metadata.
5. Run:

```bash
uv run sol-execbench examples/hip_cpp/rmsnorm \
  --solution examples/hip_cpp/rmsnorm/solution_hip.json \
  --json \
  -o out/cookbook/hip_cpp_rmsnorm.trace.jsonl
```

## Recipe: Run The Curated Slice Manually

Use this curated slice recipe for bounded v1.15 release-preview evidence.

Run a small representative subset first:

```bash
mkdir -p out/curated

uv run sol-execbench tests/sol_execbench/samples/rmsnorm \
  --solution tests/sol_execbench/samples/rmsnorm/solution_triton.json \
  --json \
  -o out/curated/rmsnorm.trace.jsonl

uv run sol-execbench examples/triton/rmsnorm \
  --solution examples/triton/rmsnorm/solution_triton.json \
  --json \
  -o out/curated/triton_rmsnorm.trace.jsonl

uv run sol-execbench examples/hip_cpp/rmsnorm \
  --solution examples/hip_cpp/rmsnorm/solution_hip.json \
  --json \
  -o out/curated/hip_cpp_rmsnorm.trace.jsonl
```

Then add ROCm library examples when dependencies are installed:

```bash
uv run sol-execbench examples/hipblas/gemm \
  --solution examples/hipblas/gemm/solution_hipblas.json \
  --json \
  -o out/curated/hipblas_gemm.trace.jsonl
```

## Recipe: Capture Environment Evidence

```bash
uv run sol-execbench doctor --json > out/cookbook/doctor.json

SOLEXECBENCH_ENV_SNAPSHOT=1 \
  uv run sol-execbench tests/sol_execbench/samples/rmsnorm \
    --solution tests/sol_execbench/samples/rmsnorm/solution_triton.json \
    --json \
    -o out/cookbook/rmsnorm.env.trace.jsonl
```

Expected sidecar:

```text
out/cookbook/rmsnorm.env.trace.jsonl.environment.json
```

## Recipe: Collect rocprofv3 Diagnostics

```bash
uv run sol-execbench tests/sol_execbench/samples/rmsnorm \
  --solution tests/sol_execbench/samples/rmsnorm/solution_triton.json \
  --profile rocprofv3 \
  --json \
  -o out/cookbook/rmsnorm.profile.trace.jsonl
```

Expected sidecars:

```text
out/cookbook/rmsnorm.profile.trace.jsonl.profile.json
out/cookbook/rmsnorm.profile.trace.jsonl.rocprofv3/
```

If `rocprofv3` is unavailable, the profile sidecar records that status and the
benchmark follows the normal path.

## Recipe: Inspect ROCm Toolchain Routing

```bash
uv run sol-execbench toolchain --json \
  --evidence-level profiling \
  --artifact-type executable_run \
  --gpu-arch gfx1200
```

Inspect the selected tool, status, fallback, and reason code. A successful
routing decision means the toolchain path is available for that evidence level;
it does not prove correctness, performance, static kernel evidence, or
leaderboard readiness.

List the built-in tool registry:

```bash
uv run sol-execbench toolchain --json --list-registry
```

## Recipe: Generate AMD-Native Derived Evidence

Use this recipe only when generating AMD-native score evidence from local
readiness, timing, and bound artifacts.

After downloading public benchmark data and creating readiness sidecars:

```bash
uv run scripts/run_dataset.py data/SOL-ExecBench/benchmark \
  --ready-subset out/ready_subset.json \
  --readiness out/readiness.json \
  --limit 5 \
  --max-workloads 1 \
  --amd-score-report out/cookbook/amd-score-report.json \
  --amd-sol-bound-dir out/cookbook/amd-sol-bounds \
  --solar-derivation out/cookbook/solar-derivation \
  --execution-closure out/cookbook/execution_closure.json
```

Treat output as AMD-native-derived evidence, not NVIDIA B200, upstream SOLAR,
leaderboard, or full paper parity.
