---
phase: 73
plan: 73-01
subsystem: static-evidence-contract
tags:
  - static-kernel-evidence
  - contract
  - guardrails
key-files:
  - src/sol_execbench/core/bench/static_kernel_evidence.py
  - src/sol_execbench/core/data/contract.py
  - tests/sol_execbench/test_static_kernel_evidence.py
  - tests/sol_execbench/test_contract.py
  - tests/sol_execbench/test_public_contract_guardrails.py
  - tests/sol_execbench/test_trace_reporting_and_score_guardrails.py
metrics:
  tests: 53 passed
---

# Phase 73 Plan 01 Summary

## Objective

Define the strict diagnostic-only static kernel evidence sidecar contract,
advertise it as optional evaluator metadata, and add CPU-safe guardrails proving
the contract stays sidecar-only.

## Completed Tasks

| Task | Result |
|------|--------|
| Task 1: Define strict static evidence sidecar models and pure helpers | Added `src/sol_execbench/core/bench/static_kernel_evidence.py` with strict frozen Pydantic models, locked status/reason enums, authority flags, stable empty-list sections, classification fields, and pure helper constructors. |
| Task 2: Advertise static evidence as an optional evaluator capability | Added `static_kernel_evidence.v1` to evaluator optional capabilities while preserving contract version `1.0` and canonical trace field groups. |
| Task 3: Add no-mutation and public-boundary guardrails | Added static-evidence exclusions to public contract guardrails and a scoring/reporting isolation regression test. |

## Commits

| Commit | Description |
|--------|-------------|
| `e98b522` | Added static evidence contract models, optional capability metadata, tests, guardrails, and restored the historical MI300X-on-CDNA3 requirements boundary string expected by guardrail coverage. |

## Verification

Commands run successfully:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_static_kernel_evidence.py tests/sol_execbench/test_contract.py tests/sol_execbench/test_public_contract_guardrails.py tests/sol_execbench/test_trace_reporting_and_score_guardrails.py -q
```

Result: `53 passed in 3.00s`

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run ruff check src/sol_execbench/core/bench/static_kernel_evidence.py src/sol_execbench/core/data/contract.py tests/sol_execbench/test_static_kernel_evidence.py tests/sol_execbench/test_contract.py tests/sol_execbench/test_public_contract_guardrails.py tests/sol_execbench/test_trace_reporting_and_score_guardrails.py
```

Result: `All checks passed!`

```bash
rg -n "subprocess|shutil\\.which|rglob|glob|write_text|open\\(" src/sol_execbench/core/bench/static_kernel_evidence.py
```

Result: no matches.

## Deviations

- The first focused pytest run exposed an existing historical guardrail that
  expected `.planning/REQUIREMENTS.md` to retain the exact phrase
  `MI300X-on-CDNA3 real-hardware validation`. The v1.17 requirements already
  preserved the boundary semantically; the exact phrase was restored in the
  out-of-scope table.
- The original executor agent stalled after writing the new static evidence
  module and tests without committing. The orchestrator closed that agent,
  reviewed its uncommitted work, completed the remaining contract/guardrail
  changes, and ran verification.

## Self-Check

PASSED.

- All `SKE-CONTRACT-01` through `SKE-CONTRACT-05` requirements are covered.
- Static evidence remains contract-only and sidecar-only.
- No artifact discovery, extractor subprocess, CLI flag, report rendering, live
  ROCm validation, RGA-rich parsing, or Triton cache capture was introduced.
