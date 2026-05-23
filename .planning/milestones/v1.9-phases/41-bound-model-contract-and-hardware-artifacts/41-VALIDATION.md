---
phase: 41
slug: bound-model-contract-and-hardware-artifacts
status: approved
nyquist_compliant: true
wave_0_complete: true
created: 2026-05-23
---

# Phase 41 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 9.0.2 |
| **Config file** | `pyproject.toml` `[tool.pytest.ini_options]` |
| **Quick run command** | `uv run pytest tests/sol_execbench/test_amd_hardware_models.py tests/sol_execbench/test_amd_sol_bounds.py tests/sol_execbench/test_public_contract_guardrails.py -x` |
| **Full suite command** | `uv run pytest tests/` |
| **Estimated runtime** | ~60 seconds for focused tests; full suite environment-dependent |

---

## Sampling Rate

- **After every task commit:** Run the task-level `<verify><automated>` command from the active plan.
- **After every plan wave:** Run `uv run pytest tests/sol_execbench/test_amd_hardware_models.py tests/sol_execbench/test_amd_sol_bounds.py tests/sol_execbench/test_amd_native_score.py tests/sol_execbench/test_public_contract_guardrails.py -x`.
- **Before `$gsd-verify-work`:** Full suite should be green where environment permits: `uv run pytest tests/`.
- **Max feedback latency:** 60 seconds for focused contract tests.

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 41-01-01 | 01 | 1 | HW-01, HW-03 | T-41-02 / T-41-03 | Packaged `gfx1200` JSON is auditable and cannot smuggle legacy `validation_status`. | unit/resource | `uv run python -c "import json; from importlib import resources; p=resources.files('sol_execbench.data.amd_hardware_models').joinpath('gfx1200.json'); payload=json.loads(p.read_text(encoding='utf-8')); assert payload['architecture']=='gfx1200'; assert 'hardware_validation_status' in payload and 'model_validation_status' in payload; assert 'validation_status' not in payload"` | ✅ W0 | ✅ green |
| 41-01-02 | 01 | 1 | HW-01, HW-02, HW-03, HW-04 | T-41-01 / T-41-03 | Packaged and external JSON share one strict parser and fail closed on misleading input. | unit | `uv run pytest tests/sol_execbench/test_amd_hardware_models.py -x` | ✅ W0 | ✅ green |
| 41-01-03 | 01 | 1 | HW-01, HW-04 | T-41-02 / T-41-03 | Built artifacts contain the packaged `gfx1200.json` resource. | packaging smoke | `uv build && uv run python -c "import zipfile; from pathlib import Path; wheels=list(Path('dist').glob('*.whl')); assert wheels; latest=max(wheels, key=lambda p: p.stat().st_mtime); names=zipfile.ZipFile(latest).namelist(); assert 'sol_execbench/data/amd_hardware_models/gfx1200.json' in names"` | ✅ W0 | ✅ green |
| 41-02-01 | 02 | 2 | HW-01, HW-03, HW-04 | T-41-05 / T-41-06 | AMD SOL compatibility loads packaged defaults and emits v2 split statuses only. | unit | `uv run pytest tests/sol_execbench/test_amd_sol_bounds.py -x` | ✅ | ✅ green |
| 41-02-02 | 02 | 2 | HW-03, DOC-01 | T-41-06 / T-41-08 | Score warnings remain conservative with split validation statuses and no Trace mutation. | unit | `uv run pytest tests/sol_execbench/test_amd_native_score.py -x` | ✅ | ✅ green |
| 41-03-01 | 03 | 3 | DOC-01 | T-41-09 / T-41-10 / T-41-11 / T-41-12 | CLI, public schemas, Trace JSONL, and validation claim boundaries remain guarded. | unit/docs | `uv run pytest tests/sol_execbench/test_public_contract_guardrails.py -x` | ✅ | ✅ green |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [x] `tests/sol_execbench/test_amd_hardware_models.py` — new unit and package-resource tests for HW-01 through HW-04.
- [x] `src/sol_execbench/data/amd_hardware_models/gfx1200.json` — packaged v2 hardware model artifact consumed by tests.
- [x] Packaging smoke command for `gfx1200.json` availability in a built artifact.

---

## Manual-Only Verifications

All Phase 41 behaviors have automated verification. Real RDNA 4 hardware validation is deferred to Phase 46; CDNA 3 / MI300X and CDNA 4 validation are outside v1.9 scope.

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all MISSING references
- [x] No watch-mode flags
- [x] Feedback latency < 60s for focused tests
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** approved 2026-05-23

## Validation Audit 2026-05-23

| Metric | Count |
|--------|-------|
| Gaps found | 0 |
| Resolved | 3 metadata updates |
| Escalated | 0 |

### Audit Notes

- Re-ran `uv run pytest tests/sol_execbench/test_amd_hardware_models.py tests/sol_execbench/test_amd_sol_bounds.py tests/sol_execbench/test_amd_native_score.py tests/sol_execbench/test_public_contract_guardrails.py -x`: 40 passed.
- Re-ran `uv build`: source distribution and wheel built successfully.
- Re-ran wheel resource smoke for `sol_execbench/data/amd_hardware_models/gfx1200.json`: passed.
- Updated stale Wave 0, task status, and approval metadata to match `41-VERIFICATION.md`.
