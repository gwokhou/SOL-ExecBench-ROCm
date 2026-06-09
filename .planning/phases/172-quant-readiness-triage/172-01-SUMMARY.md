---
phase: 172-quant-readiness-triage
plan: "01"
subsystem: readiness
tags: [quant, runtime-hints, low-precision, evidence, guardrails]
dependencies: [171]
requires: []

# Dependency graph
requires:
  - phase: 171
    provides: Quant readiness triage context and ready-state baseline
affects:
  - 173
  - 174
  - readiness
  - ready-state-classification

# Tech tracking
tech-stack:
  added: []
  patterns:
    - context-aware reference hint classifier with blocker vs lexical false-positive evidence
    - quant dtype-aware readiness split (standard/FP8/NVFP4-MXFP4)
    - explicit readiness reasons for false-positive clearance

key-files:
  modified:
    - src/sol_execbench/core/dataset/inventory.py
    - src/sol_execbench/core/dataset/readiness.py
    - tests/sol_execbench/test_dataset_inventory_readiness.py
  created: []

key-decisions:
  - "Treat only true blocker contexts (`import`, `call`, `native_source`, or true solution dependency) as hard blockers in Quant readiness."
  - "Classify Quant problems as `ready` when no true blocker remains and dtypes are non-low-precision by default."
  - "Route FP8 and NVFP4/MXFP4/Blackwell paths to hardware-evidence-needed without implying low-precision hardware authority."

patterns-established:
  - "Derive reference-runtime false-positive evidence for lexical tokens (`cublas`, `cutlass`) that appear in comments, class names, docstrings, variable names, or compatibility labels."
  - "Quant readiness now records boundary codes (`quant_rocm_compatible_reference`, `quant_cuda_false_positive_cleared`) when moving from blocker clearance to ready."

requirements-completed: [QUANT-01, QUANT-02, QUANT-03, QUANT-04]

duration: "one-shot implementation pass"
completed: 2026-06-09
---

# Phase 172: Quant Readiness Triage Summary

## Performance

- **Duration:** ~30-45 min
- **Started:** 2026-06-09
- **Completed:** 2026-06-09
- **Tasks:** 4
- **Files modified:** 3

## Accomplishments

- Reworked reference-runtime hint detection in `src/sol_execbench/core/dataset/inventory.py` to split NVIDIA/CUDA blockers from lexical false positives using context-aware matching and a new `ReferenceRuntimeHintEvidence` model.
- Added per-problem `reference_runtime_false_positive_evidence` emission for ignored lexical hints while keeping `reference_runtime_hints` blocked-only.
- Updated `src/sol_execbench/core/dataset/readiness.py` to:
  - stop blanket Quant low-precision blocking
  - add `_quant_uses_low_precision_dtype`
  - add `_quant_uses_blackwell_format`
  - route standard-dtype Quant problems to `ready` with `pytorch_compatible` when no true blocker remains
  - keep FP8 and Blackwell/NVFP4/MXFP4 pathways in `needs_hardware_evidence` with explicit reason/boundary codes.
- Added/updated CPU-safe tests in `tests/sol_execbench/test_dataset_inventory_readiness.py` for all QUANT scenarios, including false-positive evidence preservation and explicit boundary assertions (`claim_boundary.hardware_validation` and `score_authority` remain false for ready Quant cases).

## Files modified

- `src/sol_execbench/core/dataset/inventory.py`
- `src/sol_execbench/core/dataset/readiness.py`
- `tests/sol_execbench/test_dataset_inventory_readiness.py`

## Decision Notes

- No dataset source content was renamed or mutated; all compatibility semantics remain in original reference text.
- Quant readiness boundary stayed explicit:
  - `ready` indicates reference/runtime compatibility only
  - low-precision validation claims remain separate and deferred where required

---
*Phase: 172-quant-readiness-triage*
*Completed: 2026-06-09*
