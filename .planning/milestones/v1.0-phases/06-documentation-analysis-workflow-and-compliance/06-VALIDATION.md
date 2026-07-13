---
phase: 06
slug: documentation-analysis-workflow-and-compliance
status: executed
nyquist_compliant: true
wave_0_complete: true
created: 2026-05-21
validated: 2026-05-21
---

# Phase 06 — Validation Strategy

> Retroactive Nyquist validation contract reconstructed from Phase 06 plans,
> summaries, and verification evidence.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest + ruff + documentation grep |
| **Config file** | `pyproject.toml` |
| **Quick run command** | `uv run --no-sync pytest tests/sol_execbench/core/data/test_solution.py tests/sol_execbench/test_rocm_schema_build_audit.py` |
| **Full suite command** | `uv run --no-sync pytest tests/` |
| **Estimated runtime** | ~5 minutes for the full suite |

---

## Sampling Rate

- **After every task commit:** Run focused schema/audit tests or documentation grep for changed docs.
- **After every plan wave:** Run the quick command above and `uv run --no-sync ruff check src tests`.
- **Before `$gsd-verify-work`:** Run the full adapted test suite.
- **Max feedback latency:** 300 seconds for full-suite validation.

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 06-01-01 | 06-01 | 1 | DOC-01, SCFG-04 | T-06-01 | README and ROCm setup docs describe ROCm-only install, Docker, datasets, and evaluation | docs/static | `rg -n "ROCm-only|uv sync --all-groups|./scripts/run_docker.sh|download_data" README.md docs/user/rocm.md` | ✅ W0 | ✅ green |
| 06-02-01 | 06-02 | 1 | DOC-02, DOC-03 | T-06-02 | Schema, trace, and analysis docs match ROCm implementation and profiling path | unit/docs | `uv run --no-sync pytest tests/sol_execbench/core/data/test_solution.py tests/sol_execbench/test_rocm_schema_build_audit.py` | ✅ W0 | ✅ green |
| 06-03-01 | 06-03 | 2 | DOC-04, DOC-05 | T-06-03 | Compliance docs state retained attribution, unsupported NVIDIA runtimes, and known gaps | docs/static | `rg -n "Unsupported NVIDIA Runtime Features|CDNA 3 full-suite validation is deferred|Apache-2.0" docs/user/compliance.md docs/user/rocm.md README.md` | ✅ W0 | ✅ green |
| 06-03-02 | 06-03 | 2 | DOC-01, DOC-02, DOC-03, DOC-04, DOC-05 | T-06-04 | Documentation changes do not break schema/tests or linted source paths | full-suite | `uv run --no-sync pytest tests/ && uv run --no-sync ruff check src tests` | ✅ W0 | ✅ green |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [x] `README.md` — ROCm-only user entry point.
- [x] `docs/user/rocm.md` — ROCm setup and hardware notes.
- [x] `docs/user/solution.md` — ROCm solution schema.
- [x] `docs/user/trace.md` — ROCm trace schema and environment fields.
- [x] `docs/internal/analysis.md` — timing, trace review, clock locking, and `rocprofv3`.
- [x] `docs/user/compliance.md` — licensing, unsupported features, and known gaps.
- [x] `tests/sol_execbench/core/data/test_solution.py` and `tests/sol_execbench/test_rocm_schema_build_audit.py` — schema/doc consistency guardrails.

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Legal review before external redistribution | DOC-04 | Automated checks cannot replace legal review of binary redistribution obligations | Review `docs/user/compliance.md`, `pyproject.toml`, `uv.lock`, and container contents before publishing binaries/images. |

---

## Validation Audit 2026-05-21

| Metric | Count |
|--------|-------|
| Gaps found | 0 |
| Resolved | 0 |
| Escalated | 1 manual-only legal redistribution review |

---

## Validation Sign-Off

- [x] All tasks have automated verify or Wave 0 dependencies.
- [x] Sampling continuity: no 3 consecutive tasks without automated verify.
- [x] Wave 0 covers all missing references.
- [x] No watch-mode flags.
- [x] Feedback latency < 300s for full-suite validation.
- [x] `nyquist_compliant: true` set in frontmatter.

**Approval:** approved 2026-05-21
