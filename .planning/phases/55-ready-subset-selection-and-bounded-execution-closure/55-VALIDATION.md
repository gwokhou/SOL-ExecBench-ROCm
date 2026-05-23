---
phase: 55
slug: ready-subset-selection-and-bounded-execution-closure
status: draft
nyquist_compliant: true
wave_0_complete: false
created: 2026-05-23
---

# Phase 55 - Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest |
| **Config file** | `pyproject.toml` |
| **Quick run command** | `uv run pytest tests/sol_execbench/test_run_dataset_execution_closure.py -n 0 -x` |
| **Full suite command** | `uv run pytest tests/sol_execbench/test_run_dataset_execution_closure.py tests/sol_execbench/test_run_dataset_amd_score.py tests/sol_execbench/test_dataset_inventory_readiness.py tests/sol_execbench/test_public_contract_guardrails.py -n 0` |
| **Estimated runtime** | ~60 seconds |

---

## Sampling Rate

- **After every task commit:** Run `uv run pytest tests/sol_execbench/test_run_dataset_execution_closure.py -n 0 -x`
- **After every plan wave:** Run `uv run pytest tests/sol_execbench/test_run_dataset_execution_closure.py tests/sol_execbench/test_run_dataset_amd_score.py tests/sol_execbench/test_public_contract_guardrails.py -n 0`
- **Before `$gsd-verify-work`:** Full relevant suite must be green
- **Max feedback latency:** 90 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 55-01-01 | 01 | 1 | EXEC-01, EXEC-02 | runner drift | Ready-subset execution still uses `run_cli()` | fixture integration | `uv run pytest tests/sol_execbench/test_run_dataset_execution_closure.py -n 0 -x` | missing | pending |
| 55-01-02 | 01 | 1 | EXEC-01, EXEC-02 | canonical mutation | Filtered workload files are output-only | fixture integration | same | missing | pending |
| 55-01-03 | 01 | 1 | EXEC-02, EXEC-03 | hidden filters | No-ready and filtered states are explicit | fixture integration | same | missing | pending |
| 55-02-01 | 02 | 2 | EXEC-03 | evidence loss | Attempted/skipped/failure/missing-trace states are per workload | fixture integration | same | missing | pending |
| 55-02-02 | 02 | 2 | EXEC-03 | hidden blockers | Optional readiness adds not-attempted blockers | fixture integration | same | missing | pending |
| 55-02-03 | 02 | 2 | EXEC-05 | provenance gaps | Closure records command/checksum/git/config provenance | fixture integration | same | missing | pending |
| 55-03-01 | 03 | 3 | EXEC-04 | derived evidence overclaim | Existing derived sidecars are referenced and missing requested sidecars are visible | fixture integration | same | missing | pending |
| 55-03-02 | 03 | 3 | EXEC-01, EXEC-04 | public contract drift | Closure stays sidecar-only and docs avoid full-validation claims | guardrail | `uv run pytest tests/sol_execbench/test_public_contract_guardrails.py -n 0 -x` | existing | pending |
| 55-03-03 | 03 | 3 | EXEC-01..EXEC-05 | regression | Relevant runner/inventory/score/guardrail suites pass | integration | `uv run pytest tests/sol_execbench/test_run_dataset_execution_closure.py tests/sol_execbench/test_run_dataset_amd_score.py tests/sol_execbench/test_dataset_inventory_readiness.py tests/sol_execbench/test_public_contract_guardrails.py -n 0` | mixed | pending |

---

## Wave 0 Requirements

- [ ] `tests/sol_execbench/test_run_dataset_execution_closure.py` - fixtures for ready-subset execution, filtered workload materialization, closure statuses, provenance, and derived evidence refs.
- [ ] Guardrail additions in `tests/sol_execbench/test_public_contract_guardrails.py`.

---

## Manual-Only Verifications

- Real ROCm ready-subset execution is optional manual validation for this
  phase. Automated phase pass is based on fixtures with monkeypatched
  `run_cli()` and no GPU requirement.

---

## Validation Sign-Off

- [x] All tasks have automated verify commands or Wave 0 dependencies
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all missing references
- [x] No watch-mode flags
- [x] Feedback latency < 90s
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** approved 2026-05-23
