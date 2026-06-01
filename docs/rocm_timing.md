# ROCm Timing Semantics

SOL ExecBench ROCm uses an accuracy-first timing rule: choose the timer whose
measurement most accurately represents the device work being benchmarked. When
one unified timing interpretation would hide differences between PyTorch,
Triton, and HIP native sources, timing is exposed as:

```text
source_type -> timer_backend -> interpretation
```

This document defines that chimney-style contract. Profiler collection remains
a derived evidence path: it can be used by benchmark or dataset workflows
without adding fields to canonical trace JSONL.

## Source-Specific Timing Policy

| source_type | timer_backend | measured activity domain | aggregation rule | interpretation |
|-------------|---------------|--------------------------|------------------|----------------|
| `pytorch` | `pytorch_profiler` | PyTorch operator attribution with device activity cross-check | Attribute PyTorch operator regions and cross-check associated device activity | PyTorch operators can dispatch multiple HIP or ROCm library kernels, so the operator attribution is separate from raw kernel activity. |
| `triton` | `rocprofv3` | kernel activity | Aggregate post-warmup ROCm kernel activity rows for generated Triton kernels launched by the measured solution call | Triton timing is generated-kernel activity after compile/autotune warmup unless evidence explicitly says otherwise. |
| `hip_native` | `rocprofv3` | kernel activity | Aggregate ROCm kernel activity rows launched by the measured solution call | HIP C++ and ROCm library categories are interpreted as native kernel or library device work inside the timing region. |
| `mixed` | `device_events` fallback until runtime evidence is available | fallback event timing | Median HIP-backed PyTorch device event elapsed times | Mixed source timing needs runtime evidence before a profiler selection is accurate. |
| `unknown` | unsupported | unsupported | unsupported until source type is classified | No accurate timer can be selected until the source type is classified. |

## Activity Domains

**kernel activity** is device kernel execution recorded by a profiler backend
such as `rocprofv3`. It is the preferred interpretation for HIP native and
Triton-generated kernel work when the profiler evidence can be correlated to
the measured solution call.

**HIP runtime** or HIP API activity is host-side runtime call activity. It is
useful for correlation and diagnostics, but it is not automatically the same as
kernel activity duration.

**PyTorch operator attribution** links Python/PyTorch operator regions to the
device work they trigger. This matters because one PyTorch operator can dispatch
multiple kernels or ROCm library calls.

**fallback event timing** uses PyTorch ROCm's HIP-backed device event API through
the historical `torch.cuda.Event` namespace. It remains a compatibility and
fallback path, but it must not be reported as profiler-backed kernel activity.

## PyTorch ROCm Naming

PyTorch ROCm intentionally exposes AMD GPU devices through `torch.cuda`
compatibility APIs. CUDA-named PyTorch profiler activity groups are compatibility
names in this context. They do not imply that SOL ExecBench ROCm is using the
NVIDIA CUDA runtime.

## Current Benchmark Boundary

The current benchmark timing function is `time_runnable()` in
`src/sol_execbench/core/bench/timing.py`. Phase 23 does not change its execution
behavior. It defines the policy that later phases use to decide when
profiler-backed timing should replace fallback event timing.

## Profiler Evidence

Profiler-backed timing evidence is a derived methodology artifact. It must
include:

- `tool_version`: the `rocprofv3` version used to collect the trace.
- `gpu_architecture`: the AMD gfx target, such as `gfx1200`.
- `activity_domain`: the measured domain, such as kernel activity or HIP
  runtime.
- `aggregation_rule`: how parsed timing rows were combined.
- `backend`: the selected timer backend.
- `interpretation`: what the duration means for the source type.
- `parsed_rows`: normalized rows used to compute the duration.
- `fallback_reason`: why fallback event timing was used, when applicable.

Fallback evidence must be explicit. Event timing may be used when profiler
evidence is unavailable or unsupported, but it must keep its backend, fallback
reason, and interpretation separate from profiler-backed kernel activity.

Canonical trace JSONL remains the benchmark output contract. Timing policy and
profiler evidence are derived methodology artifacts unless a future phase adds
an explicit documented output path.

## Optional Profiling Artifacts

Benchmark execution can collect diagnostic profiler artifacts with:

```bash
uv run sol-execbench <problem_dir> --solution solution.json \
  --profile rocprofv3 -o traces.jsonl
```

This path is separate from benchmark timing authority. It records a
`sol_execbench.rocprofv3_profile.v1` sidecar next to the trace output, prefers
`rocpd` output for full-fidelity inspection, and registers any matching CSV
artifacts that `rocprofv3` produces. The sidecar labels whether collection
succeeded, was unavailable, or failed, and includes command provenance,
working directory, timeout, artifact paths, return code, and stdout/stderr
tails.

The profile sidecar is diagnostic-only evidence. It is not correctness authority, performance authority, timing authority, score authority, paper-parity authority, leaderboard authority, or canonical trace JSONL data.
Profiler failure must not turn an otherwise passing benchmark run into a
correctness failure. When collection falls back or is skipped, the sidecar
records an explicit fallback reason through `skipped_reason` or `failed_reason`.

## Live rocprofv3 Collection

`src/sol_execbench/core/bench/rocm_profiler.py` provides a reusable live
collection adapter for workflows that opt into profiler-backed evidence. The
adapter builds a `rocprofv3 --kernel-trace --hip-runtime-trace` command, runs
the benchmark or dataset command after the `--` separator, reads generated CSV
output from a caller-controlled evidence directory, and converts rows into a
derived `Rocprofv3TimingEvidence` payload.

The adapter follows the same source-specific timing policy:

- HIP native and Triton sources can collect `rocprofv3` kernel activity timing
  when the profiler is available.
- PyTorch sources keep PyTorch operator-attribution semantics and do not
  masquerade as raw kernel activity timing.
- Mixed or unknown sources use explicit fallback or unsupported timing evidence
  until runtime evidence is specific enough to choose a more accurate timer.

Profiler failures, missing CSV output, unsupported source types, and event
fallbacks are labeled in the collection result. Compile, autotune, warmup, or
unrelated kernel activity must be excluded by the caller's benchmark command or
explicitly labeled in the resulting evidence; it must not silently become the
reported measured latency.
