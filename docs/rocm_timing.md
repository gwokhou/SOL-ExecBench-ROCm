# ROCm Timing Semantics

SOL ExecBench ROCm uses an accuracy-first timing rule: choose the timer whose
measurement most accurately represents the device work being benchmarked. When
one unified timing interpretation would hide differences between PyTorch,
Triton, and HIP native sources, timing is exposed as:

```text
source_type -> timer_backend -> interpretation
```

This document defines that chimney-style contract. Phase 23 defines semantics
only; profiler collection and parsing are implemented separately.

## Source-Specific Timing Policy

| source_type | timer_backend | measured activity domain | aggregation rule | interpretation |
|-------------|---------------|--------------------------|------------------|----------------|
| `pytorch` | `pytorch_profiler` | PyTorch operator attribution with device activity cross-check | Attribute PyTorch operator regions and cross-check associated device activity | PyTorch operators can dispatch multiple HIP or ROCm library kernels, so the operator attribution is separate from raw kernel activity. |
| `triton` | `rocprofv3` | kernel activity | Aggregate post-warmup ROCm kernel activity rows for generated Triton kernels launched by the measured solution call | Triton timing is generated-kernel activity after compile/autotune warmup unless evidence explicitly says otherwise. |
| `hip_native` | `rocprofv3` | kernel activity | Aggregate ROCm kernel activity rows launched by the measured solution call | HIP C++ and ROCm library categories are interpreted as native kernel or library device work inside the timing region. |
| `mixed` | `device_events` fallback until runtime evidence is available | fallback event timing | Median HIP-backed PyTorch device event elapsed times | Mixed source timing needs runtime evidence before a profiler selection is accurate. |
| `unknown` | unsupported | unsupported | No aggregation | No accurate timer can be selected until the source type is classified. |

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
- `parsed timing rows`: normalized rows used to compute the duration.
- `fallback reason`: why fallback event timing was used, when applicable.

Fallback evidence must be explicit. Event timing may be used when profiler
evidence is unavailable or unsupported, but it must keep its backend, fallback
reason, and interpretation separate from profiler-backed kernel activity.

Canonical trace JSONL remains the benchmark output contract. Timing policy and
profiler evidence are derived methodology artifacts unless a future phase adds
an explicit documented output path.
