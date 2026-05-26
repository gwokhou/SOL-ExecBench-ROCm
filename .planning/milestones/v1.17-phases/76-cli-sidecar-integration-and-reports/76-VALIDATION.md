---
phase: 76
slug: cli-sidecar-integration-and-reports
status: draft
nyquist_compliant: true
wave_0_complete: true
created: 2026-05-26
---

# Phase 76 - Validation Strategy

## Commands

- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_cli_environment_snapshot.py tests/sol_execbench/test_public_contract_guardrails.py -q`
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_static_kernel_evidence.py tests/sol_execbench/test_toolchain_routing.py tests/sol_execbench/test_cli_environment_snapshot.py tests/sol_execbench/test_public_contract_guardrails.py tests/sol_execbench/test_trace_reporting_and_score_guardrails.py -q`
- `UV_CACHE_DIR=/tmp/uv-cache uv run ruff check src/sol_execbench/cli/main.py tests/sol_execbench/test_cli_environment_snapshot.py tests/sol_execbench/test_public_contract_guardrails.py`

## Coverage

| Requirement | Automated Coverage |
|-------------|--------------------|
| SKE-CLI-01 | CLI help exposes `--static-evidence` with `none|auto`; default helper path emits skipped/no sidecar for `none`. |
| SKE-CLI-02 | Path helper tests assert trace-adjacent and staging fallback names. |
| SKE-CLI-03 | Helper tests assert unsupported/nonfatal sidecars and existing trace/scoring guardrails pass. |
| SKE-CLI-04 | Sidecar payload summary test covers status, artifacts, tool runs, classification, and claim boundaries. |

**Approval:** approved 2026-05-26
