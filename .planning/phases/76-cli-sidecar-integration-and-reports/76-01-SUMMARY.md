# Plan 76-01 Summary: CLI Sidecar Integration And Reports

**Completed:** 2026-05-26  
**Status:** Complete  
**Commit:** `3072ed3 #76 - Add CLI static evidence sidecars`

## What Changed

- Added primary CLI option `--static-evidence none|auto` with default `none`.
- Added static evidence sidecar/evidence path helpers:
  - `<trace>.static-evidence.json`
  - `<trace>.static-evidence/`
  - staging fallback `static-evidence.json` and `static-evidence/`
- Integrated static evidence collection after HIP/C++ compile success and
  before staging cleanup.
- Added unsupported sidecars for non-HIP/C++ paths when `auto` is requested.
- Added JSON sidecar summary with status, artifact/tool-run counts,
  classification presence, unsupported/failed counts, and claim boundaries.
- Preserved nonfatal behavior with failed sidecars when static evidence
  collection itself raises.

## Verification

- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_cli_environment_snapshot.py tests/sol_execbench/test_public_contract_guardrails.py -q`
  - Result: 46 passed
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_static_kernel_evidence.py tests/sol_execbench/test_toolchain_routing.py tests/sol_execbench/test_cli_environment_snapshot.py tests/sol_execbench/test_public_contract_guardrails.py tests/sol_execbench/test_trace_reporting_and_score_guardrails.py -q`
  - Result: 81 passed
- `UV_CACHE_DIR=/tmp/uv-cache uv run ruff check src/sol_execbench/cli/main.py tests/sol_execbench/test_cli_environment_snapshot.py tests/sol_execbench/test_public_contract_guardrails.py`
  - Result: All checks passed

## Requirement Coverage

| Requirement | Status | Evidence |
|-------------|--------|----------|
| SKE-CLI-01 | Complete | Primary CLI exposes `--static-evidence none|auto`, defaulting to `none`. |
| SKE-CLI-02 | Complete | Tests cover trace-adjacent and staging fallback sidecar/evidence paths. |
| SKE-CLI-03 | Complete | Non-CPP and collection-failure paths produce diagnostic sidecars without throwing. |
| SKE-CLI-04 | Complete | JSON sidecar includes a compact summary section with status, artifacts, tool runs, classification, and claim boundaries. |

## Deferred

- Markdown reports and dataset-level aggregation remain out of scope.
- Documentation and live validation are Phase 77.
