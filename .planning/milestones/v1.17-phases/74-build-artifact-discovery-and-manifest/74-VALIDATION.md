---
phase: 74
slug: build-artifact-discovery-and-manifest
status: draft
nyquist_compliant: true
wave_0_complete: true
created: 2026-05-26
---

# Phase 74 - Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest |
| **Config file** | `pyproject.toml` |
| **Quick run command** | `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_static_kernel_evidence.py -q` |
| **Full suite command** | `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_static_kernel_evidence.py tests/sol_execbench/driver/test_problem_packager.py -q` |
| **Estimated runtime** | ~20 seconds |

---

## Sampling Rate

- **After every task commit:** Run `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_static_kernel_evidence.py -q`
- **After every plan wave:** Run `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_static_kernel_evidence.py tests/sol_execbench/driver/test_problem_packager.py -q`
- **Before `$gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** 30 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 74-01-01 | 01 | 1 | SKE-ARTIFACT-01 | T-74-01 | Discover only current build-root artifacts | unit | `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_static_kernel_evidence.py -q` | ✅ | ⬜ pending |
| 74-01-02 | 01 | 1 | SKE-ARTIFACT-02 | T-74-02 | Persist copies before staging cleanup | unit | `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_static_kernel_evidence.py -q` | ✅ | ⬜ pending |
| 74-01-03 | 01 | 1 | SKE-ARTIFACT-03 | T-74-03 | Manifest records path, hash, size, producer, architecture, inspectability | unit | `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_static_kernel_evidence.py -q` | ✅ | ⬜ pending |
| 74-01-04 | 01 | 1 | SKE-ARTIFACT-04 | T-74-04 | Reject global scans and symlink escapes | unit | `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_static_kernel_evidence.py -q` | ✅ | ⬜ pending |
| 74-01-05 | 01 | 1 | SKE-ARTIFACT-05 | T-74-05 | Return explicit unavailable/unsupported states | unit | `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_static_kernel_evidence.py -q` | ✅ | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

Existing Phase 73 static evidence tests and models are present. Phase 74 extends
the same test file and module, so no missing test infrastructure blocks
implementation.

---

## Manual-Only Verifications

All phase behaviors have CPU-safe automated verification. No live ROCm or manual
validation is required in Phase 74.

---

## Validation Sign-Off

- [x] All tasks have automated verification
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all existing dependencies
- [x] No watch-mode flags
- [x] Feedback latency < 30s
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** approved 2026-05-26
