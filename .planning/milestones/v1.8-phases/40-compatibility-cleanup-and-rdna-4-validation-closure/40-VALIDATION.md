---
phase: 40
slug: compatibility-cleanup-and-rdna-4-validation-closure
status: approved
nyquist_compliant: true
wave_0_complete: true
created: 2026-05-22
validated: 2026-05-22
---

# Phase 40 - Validation Strategy

## Test Infrastructure

| Property | Value |
|----------|-------|
| Framework | pytest and Ruff |
| Config file | `pyproject.toml` |
| Quick run command | `uv run pytest tests/sol_execbench/test_rocm_library_examples.py tests/sol_execbench/test_rocm_library_readiness_docs.py tests/sol_execbench/test_rocm_diagnostics_reporting.py tests/examples/test_examples.py -k "hipblas or miopen or ck or rocwmma or rocm_library or supported_library or phase_4_example or source_files or readme"` |
| Full suite command | `uv run pytest tests/sol_execbench/test_rocm_library_examples.py tests/sol_execbench/test_rocm_library_readiness_docs.py tests/sol_execbench/test_rocm_diagnostics_reporting.py tests/examples/test_examples.py -k "hipblas or miopen or ck or rocwmma or rocm_library or supported_library or phase_4_example or source_files or readme" && uv run --with ruff ruff check tests/conftest.py tests/examples/test_examples.py tests/sol_execbench/test_rocm_library_examples.py tests/sol_execbench/test_rocm_library_readiness_docs.py tests/sol_execbench/test_rocm_diagnostics_reporting.py` |
| Estimated runtime | Focused suite |

## Sampling Rate

- After every task commit: run the quick command.
- After every plan wave: run the full suite command.
- Before `$gsd-verify-work`: full suite must be green or skip only for documented missing ROCm/library development headers.
- Max feedback latency: focused pytest/Ruff latency.

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 40-01-01 | 40-01 | 1 | COMPAT-01 | docs | `uv run pytest tests/sol_execbench/test_rocm_library_readiness_docs.py -k "rocm_library or readme"` | yes | green |
| 40-01-02 | 40-01 | 1 | COMPAT-02 | docs/unit | `uv run pytest tests/sol_execbench/test_rocm_library_readiness_docs.py tests/examples/test_examples.py -k "supported_library or phase_4_example"` | yes | green |
| 40-01-03 | 40-01 | 1 | COMPAT-03 | unit/docs | `uv run pytest tests/sol_execbench/test_rocm_library_examples.py tests/sol_execbench/test_rocm_library_readiness_docs.py -k "supported_library or rocm_library"` | yes | green |
| 40-01-04 | 40-01 | 1 | COMPAT-04 | docs | `uv run pytest tests/sol_execbench/test_rocm_library_readiness_docs.py -k "readme or rocm_library"` | yes | green |
| 40-01-05 | 40-01 | 1 | RDNA4-01 | integration | `uv run pytest tests/examples/test_examples.py -k "hipblas or miopen or ck or rocwmma"` | yes | green with environment guard |
| 40-01-06 | 40-01 | 1 | RDNA4-02 | unit/docs | `uv run pytest tests/sol_execbench/test_rocm_library_examples.py tests/sol_execbench/test_rocm_library_readiness_docs.py tests/sol_execbench/test_rocm_diagnostics_reporting.py` | yes | green |
| 40-01-07 | 40-01 | 1 | RDNA4-03 | docs | `uv run pytest tests/sol_execbench/test_rocm_library_readiness_docs.py -k "readme or rocm_library"` | yes | green |

## Wave 0 Requirements

Existing infrastructure covers all phase requirements.

## Manual-Only Verifications

All phase behaviors have automated verification. Full native compile/run is environment-gated when ROCm or library development headers are absent.

## Validation Audit 2026-05-22

| Metric | Count |
|--------|-------|
| Gaps found | 0 |
| Resolved | 7 requirements mapped to existing automated checks |
| Escalated | 0 |

## Validation Sign-Off

- [x] All tasks have automated verify coverage.
- [x] Sampling continuity: no 3 consecutive tasks without automated verify.
- [x] Wave 0 covers all missing references.
- [x] No watch-mode flags.
- [x] Feedback latency is bounded by focused pytest/Ruff runtime.
- [x] `nyquist_compliant: true` set in frontmatter.

**Approval:** approved 2026-05-22
