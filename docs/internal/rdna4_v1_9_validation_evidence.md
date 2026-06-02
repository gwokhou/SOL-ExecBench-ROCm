# RDNA 4 v1.9 Validation Evidence

**Date:** 2026-05-23
**Scope:** v1.9 AMD SOL/SOLAR bound modeling completion
**Validation target:** RDNA 4 `gfx1200`

## Claim Boundary

This evidence record supports the v1.9 ROCm-port claim that AMD SOL/SOLAR bound
modeling artifacts are implemented and CPU-verified with RDNA 4-scoped hardware
model metadata. It does not claim NVIDIA B200 equivalence, upstream SOLAR
equivalence, leaderboard equivalence, CDNA3-family real-hardware validation, including MI300X (`gfx942`),
or CDNA 4 validation.

## Focused Verification Commands

The v1.9 closure uses CPU-verifiable golden tests for graph extraction,
operator estimates, v2 bound artifacts, AMD-native score integration, dataset
sidecar output, and public contract guardrails:

```bash
uv run pytest \
  tests/sol_execbench/test_amd_hardware_models.py \
  tests/sol_execbench/test_amd_bound_graph.py \
  tests/sol_execbench/test_amd_bound_estimates.py \
  tests/sol_execbench/test_amd_sol_v2.py \
  tests/sol_execbench/test_amd_native_score.py \
  tests/sol_execbench/test_run_dataset_amd_score.py \
  tests/sol_execbench/test_public_contract_guardrails.py -x
```

## Derived Sample Run Shape

A small RDNA 4-scoped derived report run should emit:

- canonical trace JSON under the dataset output directory;
- AMD SOL bound artifact v2 sidecars under `out/amd-sol-bounds`;
- AMD-native score report JSON at the caller-provided report path.

Example command:

```bash
uv run scripts/run_dataset.py data/SOL-ExecBench/benchmark \
  --limit 1 \
  --category L1 \
  --amd-score-report out/amd-score-report.json \
  --amd-sol-bound-dir out/amd-sol-bounds
```

Expected derived artifacts:

- `trace_jsonl`: remains the canonical benchmark output and is not modified by
  the v2 sidecars.
- `sol_execbench.amd_sol_bound.v2`: contains graph, rich estimate, per-op
  bound, aggregate, warning, coverage, and hardware model evidence.
- `sol_execbench.amd_native_score.v1`: references trace, SOL-bound, baseline,
  and hardware-model evidence and reports scored/unscored counts.

## Golden Coverage Inventory

The focused tests cover:

- matmul and batched matmul;
- elementwise chains and activations;
- reductions, normalization, and softmax;
- data movement, logical views, broadcast views, contiguous materialization,
  and dtype conversion;
- tuple outputs;
- unsupported operations;
- missing baseline, reference-latency fallback, provisional hardware, failed
  traces, degraded v2 bounds, and unscored v2 bounds.
