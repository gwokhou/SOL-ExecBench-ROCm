---
phase: 04
slug: rocm-library-and-example-migration
status: executed
nyquist_compliant: true
wave_0_complete: true
created: 2026-05-21
validated: 2026-05-21
---

# Phase 04 — Validation Strategy

> Retroactive Nyquist validation contract reconstructed from Phase 04 plans,
> summaries, replacement notes, and verification evidence.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest + ruff |
| **Config file** | `pyproject.toml` |
| **Quick run command** | `uv run --no-sync pytest tests/sol_execbench/test_rocm_library_examples.py tests/examples/test_examples.py -k consistency` |
| **Full suite command** | `uv run --no-sync pytest tests/examples/test_examples.py tests/sol_execbench/test_rocm_library_examples.py` |
| **Estimated runtime** | ~30 seconds for consistency tests; native example runtime covered in Phase 05 |

---

## Sampling Rate

- **After every task commit:** Run affected example/schema consistency tests.
- **After every plan wave:** Run the quick command above.
- **Before `$gsd-verify-work`:** Example consistency and replacement documentation checks must pass.
- **Max feedback latency:** 60 seconds for schema/consistency tests.

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 04-01-01 | 04-01 | 1 | LIB-01, LIB-02 | T-04-01 | PyTorch and Triton examples parse under ROCm schema and target AMD hardware | unit/static | `uv run --no-sync pytest tests/examples/test_examples.py -k consistency` | ✅ W0 | ✅ green |
| 04-02-01 | 04-02 | 1 | LIB-03 | T-04-02 | Native examples use `hip_cpp`, `.hip`, `hip_cflags`, and ROCm metadata | unit/static | `uv run --no-sync pytest tests/sol_execbench/test_rocm_library_examples.py` | ✅ W0 | ✅ green |
| 04-03-01 | 04-03 | 2 | LIB-04, LIB-05, LIB-06, LIB-07 | T-04-03 | NVIDIA library/DSL categories are replaced with ROCm fallbacks or documented alternatives | unit/static | `uv run --no-sync pytest tests/sol_execbench/test_rocm_library_examples.py` | ✅ W0 | ✅ green |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [x] `tests/examples/test_examples.py` — example metadata/source consistency and execution harness.
- [x] `tests/sol_execbench/test_rocm_library_examples.py` — ROCm library/example migration audit.
- [x] `.planning/phases/04-rocm-library-and-example-migration/04-REPLACEMENTS.md` — replacement matrix for ROCm library decisions.

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Production-quality ROCm library replacement performance | LIB-04, LIB-05, LIB-06, LIB-07 | Full CK/MIOpen/rocWMMA tuning is intentionally deferred until concrete library implementations exist | Treat fallback examples as correctness-compatible placeholders; add dedicated performance tests when ROCm-native library examples are implemented. |

---

## Validation Audit 2026-05-21

| Metric | Count |
|--------|-------|
| Gaps found | 0 |
| Resolved | 0 |
| Escalated | 1 manual-only future performance/tuning item |

---

## Validation Sign-Off

- [x] All tasks have automated verify or Wave 0 dependencies.
- [x] Sampling continuity: no 3 consecutive tasks without automated verify.
- [x] Wave 0 covers all missing references.
- [x] No watch-mode flags.
- [x] Feedback latency < 60s for focused tests.
- [x] `nyquist_compliant: true` set in frontmatter.

**Approval:** approved 2026-05-21
