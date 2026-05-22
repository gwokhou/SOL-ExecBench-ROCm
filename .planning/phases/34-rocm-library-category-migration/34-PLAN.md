# Phase 34 Plan: ROCm Library Category Migration

**Status:** Complete

## Tasks

- [x] Add a public hipBLAS-backed GEMM example with definition, workload,
  `solution_hipblas.json`, source file, `hipblasSgemm` call, and `-lhipblas`
  linker flag.
- [x] Verify the hipBLAS example parses under the ROCm schema and stages through
  the native packager.
- [x] Update library readiness docs, solution docs, and README support wording
  so `hipblas` is supported while `miopen`, `ck`, and `rocwmma` remain
  candidates.
- [x] Update public-contract tests to allow only the supported `hipblas` public
  library example and continue blocking candidate-category overclaims.
- [x] Adjust static reward-hack review so native library calls using
  `Tensor::data_ptr()` are not confused with Python semantic caches.
- [x] Add tests for legitimate native library `data_ptr` usage.

## Verification

- `uv run pytest tests/sol_execbench/test_rocm_library_examples.py tests/sol_execbench/test_rocm_library_readiness_docs.py tests/sol_execbench/core/bench/test_reward_hack.py`
- `uv run ruff check src/sol_execbench/core/bench/reward_hack.py tests/sol_execbench/core/bench/test_reward_hack.py tests/sol_execbench/test_rocm_library_examples.py tests/sol_execbench/test_rocm_library_readiness_docs.py`
