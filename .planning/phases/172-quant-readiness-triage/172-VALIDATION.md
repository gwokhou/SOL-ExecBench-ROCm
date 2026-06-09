---
phase: 172-quant-readiness-triage
slug: 172-quant-readiness-triage
status: draft
validated: 2026-06-09
nyquist_compliant: false
wave_0_complete: false
created: 2026-06-09
---

# Phase 172 — Validation Strategy

## Test Infrastructure

| Property | Value |
| --- | --- |
| Framework | pytest |
| Config file | pyproject.toml |
| Quick run command | `UV_CACHE_DIR=/tmp/sol-execbench-uv uv run pytest tests/sol_execbench/test_dataset_inventory_readiness.py -q` |
| Full suite command | `UV_CACHE_DIR=/tmp/sol-execbench-uv uv run pytest tests/sol_execbench/test_dataset_inventory_readiness.py -q` |
| Estimated runtime | ~8s |

## Sampling Rate

- **After every task commit:** quick command.
- **After every plan wave:** full command.
- **Before `$gsd-verify-work`:** run full command.
- **Max feedback latency:** ~10s.

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 172-01-01 | 01 | 1 | QUANT-01/QUANT-03 | T-172-01 | Context-aware runtime-hint classification for true blockers vs lexical false positives | unit | `UV_CACHE_DIR=/tmp/sol-execbench-uv uv run pytest tests/sol_execbench/test_dataset_inventory_readiness.py -q` | ✅ | ⚠️ partial |
| 172-01-02 | 01 | 1 | QUANT-02/QUANT-03/QUANT-04 | T-172-02 | Quant routing into ready / blocked / hardware-evidence classes | unit | `UV_CACHE_DIR=/tmp/sol-execbench-uv uv run pytest tests/sol_execbench/test_dataset_inventory_readiness.py -q` | ✅ | ⚠️ partial |
| 172-01-03 | 01 | 1 | QUANT-01..04 | T-172-03 | Regression coverage for synthetic Quant/FlashInfer/precision cases | unit | `UV_CACHE_DIR=/tmp/sol-execbench-uv uv run pytest tests/sol_execbench/test_dataset_inventory_readiness.py -q` | ✅ | ⚠️ partial |
| 172-01-04 | 01 | 1 | QUANT-01..04 | T-172-04 | Determinism and backward compatibility | integration | `UV_CACHE_DIR=/tmp/sol-execbench-uv uv run pytest tests/sol_execbench/test_dataset_inventory_readiness.py -q` | ✅ | ⚠️ partial |

### Validation note

The current test file reports 4 failing assertions: schema-input false-positive handling and FlashInfer runtime-code expectations are currently misaligned, blocking Nyquist compliance.

## Wave 0 Requirements

- [x] `tests/sol_execbench/test_dataset_inventory_readiness.py` covers all named scenarios.
- [ ] `tests/sol_execbench/test_dataset_inventory_readiness.py` fully green.

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Resolve failing expectations for Quant false-positive + FlashInfer bucket reason cases | QUANT-01..04 | Failing assertions are in shared test matrix and require code/test expectation alignment. | Fix underlying readiness/evidence behavior and rerun `UV_CACHE_DIR=/tmp/sol-execbench-uv uv run pytest tests/sol_execbench/test_dataset_inventory_readiness.py -q`; expect 0 failures. |

## Validation Audit 2026-06-09

| Metric | Count |
|--------|-------|
| Gaps found | 1 |
| Resolved | 3 |
| Escalated | 1 |

## Validation Sign-Off

- [ ] All tasks have automated verification.
- [ ] `nyquist_compliant: true` set in frontmatter.

**Approval:** pending (4 failing tests in shared readiness suite)
