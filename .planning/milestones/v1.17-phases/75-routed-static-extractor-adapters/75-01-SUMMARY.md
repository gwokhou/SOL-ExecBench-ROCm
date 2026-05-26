# Plan 75-01 Summary: Routed Static Extractor Adapters

**Completed:** 2026-05-26  
**Status:** Complete  
**Commit:** `86590cf #75 - Add routed static extractor adapters`

## What Changed

- Promoted `llvm-objdump` and `readelf` to active static toolchain routes.
- Added `raw_output_path` to static evidence tool-run records.
- Added `run_static_kernel_extractors()` for routed, bounded static extraction
  against Phase 74 persisted artifacts.
- Added bounded raw output artifacts under `extractors/<artifact-id>/<tool-id>.txt`.
- Added nonfatal unavailable, unsupported, failed, partial, and timeout
  semantics for optional extractor behavior.
- Added CPU-safe tests with injected `which`, probe runners, and extractor
  runners.

## Verification

- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_static_kernel_evidence.py tests/sol_execbench/test_toolchain_routing.py -q`
  - Result: 28 passed
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_static_kernel_evidence.py tests/sol_execbench/test_toolchain_routing.py tests/sol_execbench/test_public_contract_guardrails.py tests/sol_execbench/test_trace_reporting_and_score_guardrails.py -q`
  - Result: 65 passed
- `UV_CACHE_DIR=/tmp/uv-cache uv run ruff check src/sol_execbench/core/bench/static_kernel_evidence.py src/sol_execbench/core/toolchain.py tests/sol_execbench/test_static_kernel_evidence.py tests/sol_execbench/test_toolchain_routing.py`
  - Result: All checks passed

## Requirement Coverage

| Requirement | Status | Evidence |
|-------------|--------|----------|
| SKE-EXTRACT-01 | Complete | Extractor helper routes each tool through `build_toolchain_routing_report()`. |
| SKE-EXTRACT-02 | Complete | Tests execute bounded routed `llvm-objdump --disassemble`. |
| SKE-EXTRACT-03 | Complete | Tests execute bounded routed `readelf --headers --wide`. |
| SKE-EXTRACT-04 | Complete | Tool runs record command, timeout, return code, stdout/stderr tails, and raw output path. |
| SKE-EXTRACT-05 | Complete | Raw output artifacts are preserved; classification remains conservative. |
| SKE-EXTRACT-06 | Complete | Missing, failed, timed-out, and unsupported extractors return nonfatal sidecars. |

## Deferred

- CLI opt-in and final sidecar writing remain Phase 76.
- RGA and `roc-objdump` execution remain deferred until fixtures validate
  command behavior.
- Rich resource parsing and instruction taxonomy remain out of scope.
