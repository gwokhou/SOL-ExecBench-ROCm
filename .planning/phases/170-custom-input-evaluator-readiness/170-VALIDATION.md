---
phase: 170-custom-input-evaluator-readiness
slug: 170-custom-input-evaluator-readiness
status: draft
validated: 2026-06-09
nyquist_compliant: false
wave_0_complete: false
created: 2026-06-09
---

# Phase 170 — Validation Strategy

## Test Infrastructure

| Property | Value |
| --- | --- |
| Framework | pytest |
| Config file | pyproject.toml |
| Quick run command | `UV_CACHE_DIR=/tmp/sol-execbench-uv uv run pytest tests/sol_execbench/core/bench/test_io.py -q` |
| Full suite command | `UV_CACHE_DIR=/tmp/sol-execbench-uv uv run pytest tests/sol_execbench/core/bench/test_io.py tests/sol_execbench/driver/test_eval_driver.py tests/sol_execbench/test_dataset_inventory_readiness.py -q` |
| Estimated runtime | ~30s |

## Sampling Rate

- **After every task commit:** Task-scoped command for touched tests.
- **After every plan wave:** Full suite command above.
- **Before `$gsd-verify-work`:** run the full suite command.
- **Max feedback latency:** ~30s.

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 170-01-01 | 01 | 1 | CUST-01/CUST-03 | T-170-01 | Deterministic custom-input generation with seed/provenance | unit | `UV_CACHE_DIR=/tmp/sol-execbench-uv uv run pytest tests/sol_execbench/core/bench/test_io.py -q` | ✅ | ✅ |
| 170-01-02 | 01 | 1 | CUST-02/CUST-04 | T-170-02 | Input validation and generation-specific failure classes before runtime/candidate | unit | `UV_CACHE_DIR=/tmp/sol-execbench-uv uv run pytest tests/sol_execbench/driver/test_eval_driver.py -q` | ✅ | ✅ |
| 170-01-03 | 01 | 1 | CUST-01/CUST-02/CUST-04 | T-170-03 | Readiness reclassification without coverage recompute | integration/unit | `UV_CACHE_DIR=/tmp/sol-execbench-uv uv run pytest tests/sol_execbench/test_dataset_inventory_readiness.py -q` | ✅ | ⚠️ partial |
| 170-01-04 | 01 | 1 | CUST-01..04 | T-170-01/T-170-02/T-170-03 | CPU-safe guardrail and closure wiring evidence | integration | `UV_CACHE_DIR=/tmp/sol-execbench-uv uv run pytest tests/sol_execbench/test_e2e.py tests/sol_execbench/test_execution_closure_contract.py tests/sol_execbench/test_run_dataset_execution_closure.py -q` | ✅ | ⚠️ partial |

### Validation note

`test_dataset_inventory_readiness.py` currently has unresolved failures tied to schema-input false-positive handling and FlashInfer runtime-bucket expectation alignment, which keep this phase Nyquist open.

## Wave 0 Requirements

- [x] `tests/sol_execbench/core/bench/test_io.py` — custom input determinism and schema checks covered.
- [x] `tests/sol_execbench/driver/test_eval_driver.py` — generation-class diagnostics and propagation covered.
- [ ] `tests/sol_execbench/test_dataset_inventory_readiness.py` — full ready/reclassification branch set is not yet fully green.
- [ ] End-to-end closure guards for all CUST requirements.

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Final Phase-170 readiness split confidence when `schema_input_blocked`/quant/FlashInfer edge paths are fixed | CUST-01..04 | Shared regression file is currently blocking automation; confirm once remaining failures in `tests/sol_execbench/test_dataset_inventory_readiness.py` are resolved. | Re-run the Phase-170 validation command for `tests/sol_execbench/test_dataset_inventory_readiness.py` and confirm 0 failures. |

## Validation Audit 2026-06-09

| Metric | Count |
|--------|-------|
| Gaps found | 1 |
| Resolved | 2 |
| Escalated | 2 |

## Validation Sign-Off

- [ ] All tasks have automated verification or explicit manual-only follow-up.
- [ ] `nyquist_compliant: true` set in frontmatter.
- [ ] No watch-mode flags.
- [ ] Feedback latency < 60s.

**Approval:** pending (automation gap in shared readiness tests)
