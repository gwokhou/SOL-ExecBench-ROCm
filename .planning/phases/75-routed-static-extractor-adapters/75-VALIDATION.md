---
phase: 75
slug: routed-static-extractor-adapters
status: draft
nyquist_compliant: true
wave_0_complete: true
created: 2026-05-26
---

# Phase 75 - Validation Strategy

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest |
| **Config file** | `pyproject.toml` |
| **Quick run command** | `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_static_kernel_evidence.py tests/sol_execbench/test_toolchain_routing.py -q` |
| **Full suite command** | `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_static_kernel_evidence.py tests/sol_execbench/test_toolchain_routing.py tests/sol_execbench/test_public_contract_guardrails.py tests/sol_execbench/test_trace_reporting_and_score_guardrails.py -q` |
| **Estimated runtime** | ~25 seconds |

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Secure Behavior | Test Type | Automated Command | Status |
|---------|------|------|-------------|-----------------|-----------|-------------------|--------|
| 75-01-01 | 01 | 1 | SKE-EXTRACT-01 | Route through toolchain layer | unit | `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_toolchain_routing.py -q` | ⬜ pending |
| 75-01-02 | 01 | 1 | SKE-EXTRACT-02 | Bounded `llvm-objdump` execution | unit | `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_static_kernel_evidence.py -q` | ⬜ pending |
| 75-01-03 | 01 | 1 | SKE-EXTRACT-03 | Bounded `readelf` execution | unit | `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_static_kernel_evidence.py -q` | ⬜ pending |
| 75-01-04 | 01 | 1 | SKE-EXTRACT-04 | Command/provenance/raw-output records | unit | `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_static_kernel_evidence.py -q` | ⬜ pending |
| 75-01-05 | 01 | 1 | SKE-EXTRACT-05 | Conservative classification only | unit | `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_static_kernel_evidence.py -q` | ⬜ pending |
| 75-01-06 | 01 | 1 | SKE-EXTRACT-06 | Nonfatal unavailable/failed/timeout outcomes | unit | `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_static_kernel_evidence.py -q` | ⬜ pending |

## Manual-Only Verifications

All Phase 75 requirements have CPU-safe automated tests using fake tools and
fake outputs. Live ROCm validation remains Phase 77 work.

## Validation Sign-Off

- [x] All tasks have automated verification
- [x] No watch-mode flags
- [x] Feedback latency < 30s
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** approved 2026-05-26
