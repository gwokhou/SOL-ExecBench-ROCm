# Phase 22: RDNA 4 Validation Closure - Plan

**Status:** Executed  
**Created:** 2026-05-22  
**Requirements:** RDNA-01, RDNA-02, RDNA-03

## Objective

Validate v1.4 implementation on RDNA 4 with focused unit tests, existing E2E
tests, and one recorded `sol-execbench` CLI benchmark run.

## Scope

In scope:

- Record RDNA 4 environment evidence.
- Run v1.4 unit guardrails.
- Run existing E2E pytest coverage.
- Run `sol-execbench` CLI against an existing benchmark sample and preserve trace
  JSONL.

Out of scope:

- No CDNA 3 hardware validation.
- No new public interfaces.
- No benchmark implementation changes.

## Threat Model

| Threat | Severity | Mitigation |
|--------|----------|------------|
| T-22-01: Validation command runs on the wrong architecture. | High | Record PyTorch/ROCm `gfx1200` and `rocminfo` evidence. |
| T-22-02: E2E evidence is not durable. | Medium | Commit trace JSONL and validation evidence artifact. |
| T-22-03: CDNA 3 validation is implied. | High | Verification states CDNA 3 remains deferred. |

## Tasks

### Task 22-01: RDNA 4 Unit Validation

Run focused v1.4 unit guardrails for diagnostics, derived evidence, readiness,
support docs, and public contracts.

### Task 22-02: Existing E2E Validation

Run `uv run pytest tests/sol_execbench/test_e2e.py`.

### Task 22-03: CLI Benchmark Validation

Run the existing `sol-execbench` CLI flow against
`examples/pytorch/linear_backward` and validate emitted trace JSONL.

## Verification

All commands are recorded in `22-RDNA4-EVIDENCE.md`.
