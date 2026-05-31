# Phase 89 Plan 02 Summary: Consistency Script And Guardrails

**Status:** Complete
**Completed:** 2026-05-31

## Delivered

- Added `scripts/report_consistency.py` as a standalone sidecar report generator.
- Supported explicit input paths for execution closure, paper denominator, ROCm Matrix, runtime/static evidence, AMD score, AMD SOL/SOLAR, and AMD bound sanity reports.
- Added script coverage that writes deterministic JSON and Markdown outputs from temporary sidecar inputs.
- Extended public contract guardrails so the v1.20 consistency report remains outside canonical Definition, Workload, Trace, score, timing, and primary CLI surfaces.
- Restored v1.20 requirement wording that keeps CDNA 3, MI300X, CDNA 4, and native-host ROCm validation expansion out of scope.

## Requirements Covered

- CONS-01
- CONS-02
- CONS-03
- CONS-04
- CONS-05

## Verification

- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_consistency_report.py tests/sol_execbench/test_consistency_script.py tests/sol_execbench/test_public_contract_guardrails.py -q`
- `UV_CACHE_DIR=/tmp/uv-cache uv run ruff check src/sol_execbench/core/consistency.py scripts/report_consistency.py tests/sol_execbench/test_consistency_report.py tests/sol_execbench/test_consistency_script.py tests/sol_execbench/test_public_contract_guardrails.py`

