# Phase 22 Research: RDNA 4 Validation Closure

## RESEARCH COMPLETE

**Phase:** 22 - RDNA 4 Validation Closure  
**Date:** 2026-05-22  
**Mode:** Inline autonomous validation research

## Environment Evidence

- `rocminfo` reports a GPU agent named `gfx1200`.
- PyTorch ROCm reports:
  - `torch 2.10.0+rocm7.1`
  - `hip 7.1.25424`
  - `cuda_available True`
  - device `AMD Radeon Graphics`
  - arch `gfx1200`
- `lspci` reports `Navi 44 [Radeon RX 9060 XT]`.

## Validation Commands

Focused unit guardrails:

```bash
uv run pytest tests/sol_execbench/test_rocm_diagnostics_reporting.py tests/sol_execbench/test_trace_reporting_and_score_guardrails.py tests/sol_execbench/test_public_contract_guardrails.py tests/sol_execbench/test_rocm_support_docs.py
```

Existing E2E tests:

```bash
uv run pytest tests/sol_execbench/test_e2e.py
```

CLI benchmark E2E:

```bash
uv run sol-execbench examples/pytorch/linear_backward --solution examples/pytorch/linear_backward/solution_python.json --output .planning/phases/22-rdna-4-validation-closure/rdna4-linear-backward-traces.jsonl --json
```

## Validation Guidance

- Treat RDNA 4 as validated only for the commands run and evidence recorded in
  this phase.
- Treat CDNA 3 as readiness-only; do not claim CDNA 3 hardware validation.
- Preserve trace JSONL as canonical benchmark output.

## Validation Architecture

- Focused unit command must pass.
- Existing E2E pytest command must pass, allowing expected skips.
- CLI E2E trace JSONL must contain 3 `PASSED` records for
  `linear_backward_pytorch`.
