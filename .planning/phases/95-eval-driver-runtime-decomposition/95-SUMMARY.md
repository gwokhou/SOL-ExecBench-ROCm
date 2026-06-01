---
phase: 95
status: complete
completed: 2026-06-01
---

# Phase 95 Summary: Eval Driver Runtime Decomposition

## Completed

- Added `sol_execbench.core.bench.eval_runtime` for staged problem loading,
  entry-point parsing, reference module loading, native ROCm solution detection,
  dynamic C++ extension blocking, and user function loading.
- Replaced avoidable inline setup logic in the generated
  `driver/templates/eval_driver.py` with calls into the importable runtime
  helpers while keeping staging orchestration and trace glue in the template.
- Added focused unit tests for runtime helper behavior, including reference
  import errors, Python solution import, native ROCm missing-extension handling,
  and dynamic extension blocking.
- Kept existing generated-driver smoke tests passing to preserve status
  priority, reward-hack behavior, invalid-reference handling, runtime-error
  handling, and template syntax coverage.

## Verification

- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/core/bench/test_eval_runtime.py -q`  
  Result: 8 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/driver/test_eval_driver.py -q`  
  Result: 18 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run ruff check src/sol_execbench/core/bench/eval_runtime.py src/sol_execbench/driver/templates/eval_driver.py tests/sol_execbench/core/bench/test_eval_runtime.py`  
  Result: passed.

## Notes

This phase intentionally keeps the evaluator subprocess template as the
integration boundary. The extracted helpers cover deterministic setup and import
behavior, but correctness loops, timing, and trace emission still execute inside
the staged driver where they can observe the same subprocess context as before.
