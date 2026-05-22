---
phase: 37
slug: miopen-supported-replacement
status: approved
nyquist_compliant: true
wave_0_complete: true
created: 2026-05-22
validated: 2026-05-22
---

# Phase 37 - Validation Strategy

## Test Infrastructure

| Property | Value |
|----------|-------|
| Framework | pytest and Ruff |
| Config file | `pyproject.toml` |
| Quick run command | `uv run pytest tests/sol_execbench/test_rocm_library_examples.py tests/sol_execbench/test_rocm_library_readiness_docs.py tests/examples/test_examples.py -k "miopen or rocm_library or supported_library or phase_4_example or source_files"` |
| Full suite command | `uv run pytest tests/sol_execbench/test_rocm_library_examples.py tests/sol_execbench/test_rocm_library_readiness_docs.py tests/examples/test_examples.py -k "miopen or rocm_library or supported_library or phase_4_example or source_files" && uv run --with ruff ruff check tests/conftest.py tests/examples/test_examples.py tests/sol_execbench/test_rocm_library_examples.py` |
| Estimated runtime | Focused suite |

## Sampling Rate

- After every task commit: run the quick command.
- After every plan wave: run the full suite command.
- Before `$gsd-verify-work`: full suite must be green or skip only for documented missing ROCm development headers.
- Max feedback latency: focused pytest/Ruff latency.

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 37-01-01 | 37-01 | 1 | MIOPEN-01 | integration | `uv run pytest tests/examples/test_examples.py -k "miopen"` | yes | green with environment guard |
| 37-01-02 | 37-01 | 1 | MIOPEN-02 | unit | `uv run pytest tests/sol_execbench/test_rocm_library_examples.py -k "miopen or source_files"` | yes | green |
| 37-01-03 | 37-01 | 1 | MIOPEN-03 | unit/integration | `uv run pytest tests/sol_execbench/test_rocm_library_examples.py tests/examples/test_examples.py -k "miopen or phase_4_example"` | yes | green with environment guard |
| 37-01-04 | 37-01 | 1 | MIOPEN-04 | docs | `uv run pytest tests/sol_execbench/test_rocm_library_readiness_docs.py -k "rocm_library or supported_library"` | yes | green |

## Wave 0 Requirements

Existing infrastructure covers all phase requirements.

## Manual-Only Verifications

All phase behaviors have automated verification. Full native compile/run is environment-gated when ROCm development headers are absent.

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
- [x] Feedback latency is bounded by focused pytest/Ruff runtime.
- [x] `nyquist_compliant: true` set in frontmatter.

**Approval:** approved 2026-05-22
