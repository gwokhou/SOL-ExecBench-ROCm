# Project Research - Stack for RDNA4 Readiness Blocker Closure

## Scope

This research covers stack additions and integration points needed only for
milestone v1.34 RDNA4 Readiness Blocker Closure. Existing RDNA4 denominator,
coverage, profiler timing, clock-lock, and release evidence bundle mechanisms
are treated as established context.

## Current Stack Fit

The existing stack already has the required foundations:

- Dataset inventory and static readiness classification live under
  `src/sol_execbench/core/dataset/`.
- Execution closure and dataset runner paths already preserve denominator,
  workload, trace, timing, and blocker accounting.
- RDNA4 profiler timing coverage reports already consume readiness and timing
  sidecars and produce `coverage.json`, `coverage-summary.json`, and blocker
  ledgers.
- PyTorch ROCm is the practical semantic reference runtime for many migrated
  definitions, using PyTorch's HIP-backed `torch.cuda` namespace where needed.

## Stack Additions

### Custom Input Entrypoint Execution

Add a bounded evaluator path for benchmark-defined custom input generation:

- Resolve `custom_inputs_entrypoint` from `definition.json` or migrated
  reference source.
- Load the reference module in the existing staged/import-isolated style.
- Call the entrypoint with workload axes/scalars and the target device.
- Validate generated keys, tensor/scalar types, dtypes, shapes, and device
  placement before running reference or candidate code.
- Seed deterministically per workload so reference/candidate comparisons and
  reruns are reproducible.

No random-substitution path should be added. The blocker exists specifically
because random tensors would break benchmark semantics.

### Quant Readiness Triage

Add a safer static/runtime split for Quant readiness:

- Static hints should distinguish real CUDA runtime dependencies from comments,
  variable names, class names, and compatibility labels.
- CPU/PyTorch semantic references can be accepted when no CUDA import, native
  CUDA source, or NVIDIA-only library call is required.
- Low-precision behavior still needs explicit ROCm hardware evidence before
  validation or performance claims.

Candidate ROCm reference paths:

- PyTorch ROCm semantic reference for dequantize-then-matmul and scale metadata.
- HIP/hipBLASLt or Composable Kernel only when a performance candidate is needed.
- Triton ROCm only after semantic reference parity is established.

### FlashInfer Readiness Split

FlashInfer-Bench should not remain all-or-nothing:

- Simple PyTorch-expressible workloads can be reclassified independently.
- True FlashInfer runtime workloads need a dedicated compatibility layer for
  paged/ragged decode and prefill semantics, page tables, KV-cache layout,
  workspace assumptions, and dtype constraints.
- FlashInfer-specific performance kernels are not required for readiness
  closure unless a workload's semantics cannot be represented safely otherwise.

## Non-Additions

- Do not add NVIDIA/CUDA fallback execution paths.
- Do not redistribute benchmark dataset payloads.
- Do not change public SOL ExecBench schemas unless a ROCm-specific internal
  sidecar is required.
- Do not upgrade CDNA3 or CDNA4 claims from RDNA4 readiness evidence.

## Integration Points

| Capability | Primary Integration | Supporting Files |
| --- | --- | --- |
| Custom input generation | Dataset/eval driver input assembly | `core/dataset/readiness.py`, eval driver helpers, runner tests |
| Quant hint triage | Inventory/readiness classifier | `core/dataset/readiness.py`, inventory runtime hints |
| FlashInfer split | Migration/readiness classifier | FlashInfer migration helpers, readiness tests |
| Coverage recompute | Profiler timing coverage reports | `core/dataset/profiler_timing_coverage.py`, RDNA4 scripts |
| Claim guardrails | Docs and public contract tests | `docs/CLAIMS.md`, guardrail tests |

