# Phase 35 Context: MI300X Validation Readiness Guardrails

**Date:** 2026-05-22
**Status:** Complete

## Problem

The project supports CDNA 3 schema/build targets but has not run on real
MI300X hardware. The milestone needed a concrete future validation handoff,
explicit FP8/NVFP4 decisions, and a guard that prevents reports from upgrading
MI300X-on-CDNA3 status without recorded full-suite evidence.

## Relevant Code

- `src/sol_execbench/core/diagnostics.py` contains ROCm readiness helpers.
- `docs/internal/cdna3_validation_readiness.md` documents existing CDNA 3
  readiness.
- `README.md`, `docs/rocm.md`, and `.planning/REQUIREMENTS.md` carry public
  no-claim language.
- `tests/sol_execbench/test_rocm_diagnostics_reporting.py` and
  `tests/sol_execbench/test_rocm_support_docs.py` protect readiness behavior and
  documentation.

## Constraints

- Do not require real MI300X hardware during this phase.
- Do not mark MI300X-on-CDNA3 as hardware-validated.
- FP8 is future MI300X validation scope; NVFP4/MXFP4 remains deferred.
