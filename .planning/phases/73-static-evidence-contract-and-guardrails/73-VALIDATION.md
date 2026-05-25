---
phase: 73
slug: static-evidence-contract-and-guardrails
status: draft
nyquist_compliant: true
wave_0_complete: false
created: 2026-05-25
---

# Phase 73 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest |
| **Config file** | `pyproject.toml` |
| **Quick run command** | `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_static_kernel_evidence.py tests/sol_execbench/test_contract.py -q` |
| **Full suite command** | `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_static_kernel_evidence.py tests/sol_execbench/test_contract.py tests/sol_execbench/test_public_contract_guardrails.py tests/sol_execbench/test_trace_reporting_and_score_guardrails.py -q` |
| **Estimated runtime** | ~20 seconds |

---

## Sampling Rate

- **After every task commit:** Run `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_static_kernel_evidence.py tests/sol_execbench/test_contract.py -q`
- **After every plan wave:** Run `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_static_kernel_evidence.py tests/sol_execbench/test_contract.py tests/sol_execbench/test_public_contract_guardrails.py tests/sol_execbench/test_trace_reporting_and_score_guardrails.py -q`
- **Before `$gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** 30 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 73-01-01 | 01 | 0 | SKE-CONTRACT-01 | — | Reject malformed or unknown sidecar fields | unit | `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_static_kernel_evidence.py -q` | ❌ W0 | ⬜ pending |
| 73-01-02 | 01 | 1 | SKE-CONTRACT-02 | — | Preserve diagnostic-only authority boundaries | unit | `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_static_kernel_evidence.py -q` | ❌ W0 | ⬜ pending |
| 73-01-03 | 01 | 1 | SKE-CONTRACT-03 | — | Serialize stable status and reason-code outcomes | unit | `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_static_kernel_evidence.py -q` | ❌ W0 | ⬜ pending |
| 73-01-04 | 01 | 1 | SKE-CONTRACT-04 | — | Add optional capability without contract bump | unit | `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_contract.py -q` | ✅ | ⬜ pending |
| 73-01-05 | 01 | 1 | SKE-CONTRACT-05 | — | Keep trace, scoring, and default CLI behavior isolated | regression | `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_public_contract_guardrails.py tests/sol_execbench/test_trace_reporting_and_score_guardrails.py -q` | ✅ | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/sol_execbench/test_static_kernel_evidence.py` — schema, authority, status, reason-code, and helper constructor tests for SKE-CONTRACT-01..03.
- [ ] Confirm existing `tests/sol_execbench/test_contract.py` coverage is extended for SKE-CONTRACT-04.
- [ ] Confirm existing public contract and score guardrail tests are extended for SKE-CONTRACT-05.

---

## Manual-Only Verifications

All phase behaviors have automated verification. No live ROCm or manual
validation is required in Phase 73.

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all MISSING references
- [x] No watch-mode flags
- [x] Feedback latency < 30s
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** approved 2026-05-25
