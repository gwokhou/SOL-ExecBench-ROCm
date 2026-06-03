---
status: passed
phase: 128
phase_name: MI300X Evidence Contract and Validation Handoff
verified_at: 2026-06-04
---

# Phase 128 Verification

## Result

Passed.

## Must-Haves

- Future validator can follow a MI300X command/evidence sequence.
  - Verified through updated `.planning/milestones/MI300X-VALIDATION-HANDOFF.md`
    and `docs/internal/mi300x_validation_readiness.md`.
- Required artifacts are programmatically enforced before a validation claim.
  - Verified by `MI300X_REQUIRED_ARTIFACTS` and diagnostics tests.
- Result categories distinguish expected skips, missing tools, functional
  failures, timing instability, missing evidence, FP8, and deferred quantized
  formats.
  - Verified by `MI300X_VALIDATION_RESULT_CATEGORIES` tests and handoff docs.
- FP8 readiness and NVFP4/MXFP4 deferral remain explicit.
  - Verified by diagnostics and support-doc tests.

## Commands

```bash
uv run pytest tests/sol_execbench/test_rocm_diagnostics_reporting.py tests/sol_execbench/test_rocm_support_docs.py
uv run --with ruff ruff check src/sol_execbench/core/diagnostics.py tests/sol_execbench/test_rocm_diagnostics_reporting.py tests/sol_execbench/test_rocm_support_docs.py
```

## Notes

This phase verifies the contract and guardrails only. It does not produce real
MI300X hardware evidence.
