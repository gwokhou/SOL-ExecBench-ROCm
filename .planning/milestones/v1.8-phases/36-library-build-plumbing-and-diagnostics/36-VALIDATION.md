---
phase: 36
slug: library-build-plumbing-and-diagnostics
status: approved
nyquist_compliant: true
wave_0_complete: true
created: 2026-05-22
validated: 2026-05-22
---

# Phase 36 - Validation Strategy

## Test Infrastructure

| Property | Value |
|----------|-------|
| Framework | pytest |
| Config file | `pyproject.toml` |
| Quick run command | `uv run pytest tests/sol_execbench/test_rocm_diagnostics_reporting.py tests/sol_execbench/test_rocm_library_examples.py tests/sol_execbench/test_rocm_library_readiness_docs.py` |
| Full suite command | `uv run pytest tests/sol_execbench/test_rocm_diagnostics_reporting.py tests/sol_execbench/test_rocm_library_examples.py tests/sol_execbench/test_rocm_library_readiness_docs.py && bash -n docker/entrypoint.sh` |
| Estimated runtime | Focused suite |

## Sampling Rate

- After every task commit: run the quick command.
- After every plan wave: run the full suite command.
- Before `$gsd-verify-work`: full suite must be green.
- Max feedback latency: focused pytest latency.

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 36-01-01 | 36-01 | 1 | BUILD-01 | unit | `uv run pytest tests/sol_execbench/test_rocm_diagnostics_reporting.py` | yes | green |
| 36-01-02 | 36-01 | 1 | BUILD-02 | unit | `uv run pytest tests/sol_execbench/test_rocm_library_examples.py` | yes | green |
| 36-01-03 | 36-01 | 1 | BUILD-03 | docs | `uv run pytest tests/sol_execbench/test_rocm_library_readiness_docs.py` | yes | green |
| 36-01-04 | 36-01 | 1 | BUILD-04 | unit | `uv run pytest tests/sol_execbench/test_rocm_library_examples.py` | yes | green |
| 36-01-05 | 36-01 | 1 | BUILD-03 | syntax | `bash -n docker/entrypoint.sh` | yes | green |

## Wave 0 Requirements

Existing infrastructure covers all phase requirements.

## Manual-Only Verifications

All phase behaviors have automated verification.

## Validation Audit 2026-05-22

| Metric | Count |
|--------|-------|
| Gaps found | 0 |
| Resolved | 4 requirements mapped to existing automated checks |
| Escalated | 0 |

## Validation Sign-Off

- [x] All tasks have automated verify coverage.
- [x] Sampling continuity: no 3 consecutive tasks without automated verify.
- [x] Wave 0 covers all missing references.
- [x] No watch-mode flags.
- [x] Feedback latency is bounded by focused pytest runtime.
- [x] `nyquist_compliant: true` set in frontmatter.

**Approval:** approved 2026-05-22
