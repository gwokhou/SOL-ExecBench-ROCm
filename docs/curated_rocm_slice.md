# Curated ROCm Benchmark Slice

The curated ROCm slice is a small research-preview surface for exercising the
port end to end. It is intentionally bounded: it is not the full paper dataset,
not full 235-problem validation, and not leaderboard parity.

## Selection Criteria

Problems belong in the slice when they satisfy all required criteria:

- They use repository-local examples or downloaded public dataset problems with
  stable `definition.json`, `workload.jsonl`, and solution metadata.
- They exercise an important solution path: PyTorch ROCm, Triton ROCm, HIP/C++,
  or a ROCm native library category.
- They can run through the existing `sol-execbench` CLI or
  `scripts/run_dataset.py`.
- Their evidence expectations can be stated clearly, including skipped,
  unavailable, degraded, or unscored states.

Problems are excluded when they need unsupported NVIDIA-only runtime paths,
missing proprietary data, unvalidated hardware, or unclear score interpretation.

## Initial Slice

| Path | Category | Purpose | Expected evidence |
| --- | --- | --- | --- |
| `tests/sol_execbench/samples/rmsnorm` | PyTorch / Triton-compatible sample | Small normalization kernel for fast local checks. | Trace JSONL; optional environment sidecar. |
| `tests/sol_execbench/samples/linear_backward` | PyTorch sample | Backward-style workload coverage. | Trace JSONL; baseline comparison if a prior trace exists. |
| `examples/triton/rmsnorm` | Triton ROCm | Representative Triton kernel path. | Trace JSONL; optional timing/profile evidence when ROCm device is visible. |
| `examples/triton/nemotron_rms_norm` | Triton ROCm | Model-shaped normalization variant. | Trace JSONL; readiness to diagnose compilation/runtime failures. |
| `examples/hip_cpp/rmsnorm` | HIP/C++ | Native extension build and execution path. | Trace JSONL; HIP build logs on failure. |
| `examples/hip_cpp/flux_rope` | HIP/C++ | Native source with model-style tensor movement. | Trace JSONL; reward-hack and build-path coverage. |
| `examples/hipblas/gemm` | ROCm library | hipBLAS native library replacement path. | Trace JSONL or explicit dependency-unavailable state. |
| `examples/ck/gemm` | ROCm library | Composable Kernel replacement path. | Trace JSONL or explicit header/dependency-unavailable state. |
| `examples/rocwmma/gemm` | ROCm library | Matrix-core library replacement path. | Trace JSONL or explicit architecture/dependency-unavailable state. |
| `examples/miopen/softmax` | ROCm library | MIOpen replacement path for cuDNN-style coverage. | Trace JSONL or explicit dependency-unavailable state. |

This list is a release-preview seed. A phase may shrink it if hardware or
dependency availability makes a smaller slice more reproducible, but any
shrinking must keep the selection criteria and excluded categories visible.

## Single-Problem Commands

Fast sample:

```bash
uv run sol-execbench tests/sol_execbench/samples/rmsnorm \
  --solution tests/sol_execbench/samples/rmsnorm/solution_triton.json \
  --json \
  -o out/curated/rmsnorm.trace.jsonl
```

HIP/C++ example:

```bash
uv run sol-execbench examples/hip_cpp/rmsnorm \
  --solution examples/hip_cpp/rmsnorm/solution_hip_cpp.json \
  --json \
  -o out/curated/hip_cpp_rmsnorm.trace.jsonl
```

ROCm library example:

```bash
uv run sol-execbench examples/hipblas/gemm \
  --solution examples/hipblas/gemm/solution_hipblas.json \
  --json \
  -o out/curated/hipblas_gemm.trace.jsonl
```

## Optional Evidence

Environment evidence:

```bash
SOLEXECBENCH_ENV_SNAPSHOT=1 \
  uv run sol-execbench tests/sol_execbench/samples/rmsnorm \
    --solution tests/sol_execbench/samples/rmsnorm/solution_triton.json \
    --json \
    -o out/curated/rmsnorm.trace.jsonl
```

Profiling evidence:

```bash
uv run sol-execbench tests/sol_execbench/samples/rmsnorm \
  --solution tests/sol_execbench/samples/rmsnorm/solution_triton.json \
  --profile rocprofv3 \
  --json \
  -o out/curated/rmsnorm.profiled.trace.jsonl
```

AMD-native score evidence, when dataset sidecars and bound artifacts are
available:

```bash
uv run scripts/run_dataset.py data/SOL-ExecBench/benchmark \
  --ready-subset out/ready_subset.json \
  --readiness out/readiness.json \
  --limit 5 \
  --max-workloads 1 \
  --amd-score-report out/curated/amd-score-report.json \
  --amd-sol-bound-dir out/curated/amd-sol-bounds \
  --solar-derivation out/curated/solar-derivation \
  --execution-closure out/curated/execution_closure.json
```

## Artifact Expectations

| Artifact | Meaning | Claim boundary |
| --- | --- | --- |
| `*.trace.jsonl` | Canonical benchmark output. | ROCm-port evidence only. |
| `*.environment.json` | Optional runtime reproducibility sidecar. | Runtime evidence, not score authority. |
| `*.profile.json` and `.rocprofv3/` | Optional profiler sidecars and artifacts. | Diagnostic evidence, not correctness or score authority. |
| AMD SOL sidecars | Derived AMD bound inputs. | AMD-native-derived evidence only. |
| AMD score report | Guarded local score interpretation. | Not B200, SOLAR, or leaderboard parity. |
| execution closure | Bounded run accounting. | Not full 235-problem validation. |

