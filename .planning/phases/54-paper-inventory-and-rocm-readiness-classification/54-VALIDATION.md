---
phase: 54
slug: paper-inventory-and-rocm-readiness-classification
status: draft
nyquist_compliant: true
wave_0_complete: false
created: 2026-05-23
---

# Phase 54 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest |
| **Config file** | `pyproject.toml` |
| **Quick run command** | `uv run pytest tests/sol_execbench/test_dataset_inventory_readiness.py -n 0 -x` |
| **Full suite command** | `uv run pytest tests/sol_execbench/test_dataset_contract.py tests/sol_execbench/test_download_solexecbench.py tests/sol_execbench/test_dataset_inventory_readiness.py tests/sol_execbench/test_public_contract_guardrails.py -n 0` |
| **Estimated runtime** | ~45 seconds |

---

## Sampling Rate

- **After every task commit:** Run `uv run pytest tests/sol_execbench/test_dataset_inventory_readiness.py -n 0 -x`
- **After every plan wave:** Run `uv run pytest tests/sol_execbench/test_dataset_contract.py tests/sol_execbench/test_download_solexecbench.py tests/sol_execbench/test_dataset_inventory_readiness.py tests/sol_execbench/test_public_contract_guardrails.py -n 0`
- **Before `$gsd-verify-work`:** Full relevant suite must be green
- **Max feedback latency:** 60 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 54-01-01 | 01 | 1 | INV-01, INV-05 | schema confusion | Inventory models serialize deterministically and use canonical schemas | unit | `uv run pytest tests/sol_execbench/test_dataset_inventory_readiness.py -n 0 -x` | missing | pending |
| 54-01-02 | 01 | 1 | INV-01, INV-04 | schema confusion | Schema failures are records, not suite aborts | unit | same | missing | pending |
| 54-01-03 | 01 | 1 | INV-02, INV-03 | claim overreach | Metadata/hints are recorded without readiness claims | unit | same | missing | pending |
| 54-02-01 | 02 | 2 | READY-01, READY-02 | misleading readiness | Workload statuses aggregate deterministically | unit | same | missing | pending |
| 54-02-02 | 02 | 2 | READY-03 | asset substitution | Custom/safetensors blockers are explicit | unit | same | missing | pending |
| 54-02-03 | 02 | 2 | READY-04 | hardware overclaim | Low-precision/Quant layers need hardware evidence | unit | same | missing | pending |
| 54-03-01 | 03 | 3 | READY-05 | canonical mutation | Ready subset is sidecar-only | unit | same | missing | pending |
| 54-03-02 | 03 | 3 | INV-01, READY-01 | CLI overreach | Thin CLI writes sidecars only | unit | same | missing | pending |
| 54-03-03 | 03 | 3 | INV-01..READY-05 | public contract drift | Guardrails protect canonical schemas and primary CLI | guardrail | `uv run pytest tests/sol_execbench/test_dataset_inventory_readiness.py tests/sol_execbench/test_public_contract_guardrails.py -n 0 -x` | existing | pending |

---

## Wave 0 Requirements

- [ ] `tests/sol_execbench/test_dataset_inventory_readiness.py` — fixtures for inventory, readiness, ready subset, and CLI behavior.
- [ ] Guardrail additions in `tests/sol_execbench/test_public_contract_guardrails.py`.

---

## Manual-Only Verifications

All phase behaviors have automated verification. Real GPU execution and ready
subset execution are intentionally deferred to Phase 55.

---

## Validation Sign-Off

- [x] All tasks have automated verify commands or Wave 0 dependencies
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all missing references
- [x] No watch-mode flags
- [x] Feedback latency < 60s
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** approved 2026-05-23
