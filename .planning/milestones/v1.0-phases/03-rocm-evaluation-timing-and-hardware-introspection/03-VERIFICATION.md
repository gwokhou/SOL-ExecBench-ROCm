---
phase: 03-rocm-evaluation-timing-and-hardware-introspection
status: passed
verified_at: 2026-05-21
---

# Phase 03 Verification

## Result

PASSED.

## Requirement Coverage

| Requirement | Result | Evidence |
| --- | --- | --- |
| EVAL-01 | PASS | `eval_driver.py` routes `HIP_CPP`, `HIPBLAS`, `MIOPEN`, `CK`, and `ROCWMMA` through `benchmark_kernel.so`; Python paths remain for PyTorch and Triton. |
| EVAL-02 | PASS | Existing eval-driver tests still pass destination-passing and return-value behavior through trace emission. |
| EVAL-03 | PASS | Input generation, output normalization, correctness checks, and strict JSONL trace emission remained covered by `test_eval_driver.py`. |
| EVAL-04 | PASS | `timing.py` no longer imports or calls CUPTI; default timing uses ROCm-compatible PyTorch device events with setup exclusion and post-call synchronization. |
| EVAL-05 | PASS | Focused audit blocks legacy native CUDA enum/schema values in Phase 3 runtime paths. |
| PROF-01 | PASS | Environment snapshots report `hip` and `rocm` when PyTorch exposes `torch.version.hip`. |
| PROF-02 | PASS | Clock-lock implementation uses `rocm-smi`, not `nvidia-smi`. |
| PROF-03 | PASS | Clock verification checks active ROCm SCLK/MCLK levels and returns false on missing/failed tooling. |
| PROF-04 | PASS | `lock_clocks=True` still rejects workloads before timing if `SOL_EXECBENCH_CLOCKS_LOCKED` is not true. |
| PROF-05 | PASS | Phase 3 audit protects ROCm timing and hardware-introspection paths from CUDA/NVIDIA tooling regressions. |

## Tests

- `uv run --no-sync pytest tests/sol_execbench/driver/test_eval_driver.py` -> 15 passed.
- `uv run --no-sync pytest tests/sol_execbench/core/bench/test_timing.py` -> 57 skipped under current marker configuration.
- `uv run --no-sync pytest tests/sol_execbench/core/bench/test_clock_lock.py tests/sol_execbench/driver/test_eval_driver.py` -> 41 passed.
- `uv run --no-sync pytest tests/sol_execbench/test_rocm_eval_timing_audit.py` -> 3 passed.
- `uv run --no-sync pytest tests/sol_execbench/driver/test_eval_driver.py tests/sol_execbench/core/bench/test_timing.py tests/sol_execbench/core/bench/test_clock_lock.py tests/sol_execbench/test_rocm_eval_timing_audit.py` -> 44 passed, 57 skipped.
- `uv run --no-sync ruff check ...` for all changed Python files -> passed.

## Residual Risk

No ROCm GPU timing integration was executed in this environment because timing tests were marker-skipped. Phase 5 owns RDNA4/CDNA3 hardware matrix validation.
