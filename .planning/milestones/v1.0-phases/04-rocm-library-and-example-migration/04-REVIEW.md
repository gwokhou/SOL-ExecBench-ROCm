---
phase: 04-rocm-library-and-example-migration
status: passed
reviewed_at: 2026-05-21
---

# Phase 04 Code Review

## Findings

No open blocking findings.

## Fixed During Review

- Avoided speculative mechanical rewrites for CUTLASS/cuDNN/CuTe/cuTile examples by replacing them with explicit PyTorch fallback examples and documenting ROCm replacement paths.
- Synchronized embedded `sources[].content` fields with public example source files after HIP file renames.
- Updated example test descriptors to recognize ROCm native language values.

## Residual Risk

- Public example e2e execution was not run because GPU execution and marker overhaul are Phase 5 responsibilities.
- HIP/C++ example kernels are schema- and source-consistency migrated, but hardware compilation remains to be validated in Phase 5.

## Verification

- `uv run --no-sync pytest tests/examples/test_examples.py -k consistency tests/sol_execbench/test_rocm_library_examples.py` -> 11 passed.
- `uv run --no-sync ruff check examples/cudnn/softmax/kernel.py examples/cutlass/gemm/kernel.py examples/cute_dsl/jamba_attn_proj/kernel.py examples/cutile/jamba_attn_proj/kernel.py tests/examples/test_examples.py tests/sol_execbench/test_rocm_library_examples.py` -> passed.
