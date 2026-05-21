---
phase: 04-rocm-library-and-example-migration
plan: 02
subsystem: library-replacements
tags:
  - rocm
  - examples
  - replacements
key-files:
  - examples/cutlass/gemm/solution_cutlass.json
  - examples/cudnn/softmax/solution_cudnn.json
  - examples/cute_dsl/jamba_attn_proj/solution_cute_dsl.json
  - examples/cutile/jamba_attn_proj/solution_cutile.json
  - .planning/phases/04-rocm-library-and-example-migration/04-REPLACEMENTS.md
metrics:
  tests: "11 passed"
---

# Plan 04-02 Summary

## Changes

- Replaced former CUTLASS, cuDNN, CuTe DSL, and cuTile example solutions with ROCm-schema-compatible PyTorch fallback examples.
- Removed stale public `.cu` and NVIDIA library source files from the affected example directories.
- Added `04-REPLACEMENTS.md` documenting pragmatic replacement choices for rocBLAS, hipBLASLt, MIOpen, Composable Kernel, rocWMMA, hipCUB, rocPRIM, and rocThrust.
- Updated jamba sample solution metadata to fallback PyTorch implementations instead of rejected NVIDIA DSL languages.

## Verification

- `uv run --no-sync pytest tests/examples/test_examples.py -k consistency tests/sol_execbench/test_rocm_library_examples.py` -> 11 passed.

## Deviations

- Full CK/MIOpen/rocWMMA implementations were not created in this phase because they require hardware validation and shape-specific tuning. This follows the user-selected pragmatic migration strategy.

## Self-Check: PASSED

NVIDIA-specific library/DSL examples no longer use rejected schema values, and replacement rationale is documented.
