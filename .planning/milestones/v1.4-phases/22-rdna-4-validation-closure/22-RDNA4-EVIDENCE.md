# Phase 22 RDNA 4 Validation Evidence

**Date:** 2026-05-22  
**Host GPU:** AMD Radeon Graphics  
**Architecture:** `gfx1200`  
**PCI Device:** Navi 44 [Radeon RX 9060 XT]  
**Torch:** `2.10.0+rocm7.1`  
**HIP:** `7.1.25424`

## Environment Checks

`rocminfo` reported GPU agent `gfx1200` with marketing name `AMD Radeon
Graphics`.

PyTorch ROCm check:

```text
torch 2.10.0+rocm7.1
hip 7.1.25424
cuda_available True
device AMD Radeon Graphics
arch gfx1200
```

## Focused Unit Validation

```bash
uv run pytest tests/sol_execbench/test_rocm_diagnostics_reporting.py tests/sol_execbench/test_trace_reporting_and_score_guardrails.py tests/sol_execbench/test_public_contract_guardrails.py tests/sol_execbench/test_rocm_support_docs.py
```

Result:

```text
25 passed in 53.49s
```

## Existing E2E Validation

```bash
uv run pytest tests/sol_execbench/test_e2e.py
```

Result:

```text
5 passed, 1 skipped in 61.53s
```

## CLI Benchmark E2E

```bash
uv run sol-execbench examples/pytorch/linear_backward --solution examples/pytorch/linear_backward/solution_python.json --output .planning/phases/22-rdna-4-validation-closure/rdna4-linear-backward-traces.jsonl --json
```

Result:

- Problem: `linear_backward`
- Solution: `linear_backward_pytorch`
- Workloads: 3
- Trace file: `.planning/phases/22-rdna-4-validation-closure/rdna4-linear-backward-traces.jsonl`
- Statuses: `PASSED`, `PASSED`, `PASSED`
- Environment in trace:
  - hardware: `AMD Radeon Graphics`
  - torch: `2.10.0+rocm7.1`
  - triton: `3.6.0`
  - hip/rocm: `7.1.25424`

## Claim Boundary

This validates the v1.4 path on the visible RDNA 4 `gfx1200` environment. It
does not validate CDNA 3 hardware.
