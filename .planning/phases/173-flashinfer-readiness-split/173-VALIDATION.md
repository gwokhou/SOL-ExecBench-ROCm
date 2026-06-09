---
phase: 173-flashinfer-readiness-split
slug: 173-flashinfer-readiness-split
status: draft
validated: 2026-06-09
nyquist_compliant: false
wave_0_complete: false
created: 2026-06-09
---

# Phase 173 — Validation Strategy

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
- **Before `$gsd-verify-work`:** quick command after any readiness classifier changes.
- **Max feedback latency:** ~10s.

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 173-01-01 | 01 | 1 | FLASH-01/FLASH-02/FLASH-03/FLASH-04 | T-173-01 | Semantic taxonomy and runtime-bucket classification | unit | `UV_CACHE_DIR=/tmp/sol-execbench-uv uv run pytest tests/sol_execbench/test_dataset_inventory_readiness.py -q` | ✅ | ⚠️ partial |
| 173-01-02 | 01 | 1 | FLASH-01/FLASH-02/FLASH-03/FLASH-04 | T-173-01 | Classifier regression for simple/runtime examples | unit | `UV_CACHE_DIR=/tmp/sol-execbench-uv uv run pytest tests/sol_execbench/test_dataset_inventory_readiness.py -q` | ✅ | ⚠️ partial |

### Validation note

`test_readiness_classifies_flashinfer_semantic_buckets` currently fails on one runtime-bucket code expectation; unresolved expectation drift blocks full Nyquist.

## Wave 0 Requirements

- [x] Coverage tests for FlashInfer semantic buckets exist in `tests/sol_execbench/test_dataset_inventory_readiness.py`.
- [ ] All assertions pass.

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Finalize runtime reason mappings and expected-code stability (`flashinfer_runtime_*`) | FLASH-03 | Classifier behavior mismatch indicates semantic bucket assertion gap. | Align expected runtime reason mapping in implementation/tests and rerun `UV_CACHE_DIR=/tmp/sol-execbench-uv uv run pytest tests/sol_execbench/test_dataset_inventory_readiness.py -q`; expect 0 failures. |

## Validation Audit 2026-06-09

| Metric | Count |
|--------|-------|
| Gaps found | 1 |
| Resolved | 1 |
| Escalated | 1 |

## Validation Sign-Off

- [ ] All tasks have automated verification.
- [ ] `nyquist_compliant: true` set in frontmatter.

**Approval:** pending (1 failing test)
