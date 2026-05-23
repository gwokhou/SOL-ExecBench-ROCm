---
phase: 53
slug: dataset-contract-and-acquisition-metadata
status: draft
nyquist_compliant: true
wave_0_complete: false
created: 2026-05-23
---

# Phase 53 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest |
| **Config file** | `pyproject.toml` |
| **Quick run command** | `uv run pytest tests/sol_execbench/test_dataset_contract.py tests/sol_execbench/test_download_solexecbench.py -x` |
| **Full suite command** | `uv run pytest tests/sol_execbench/test_dataset_contract.py tests/sol_execbench/test_download_solexecbench.py tests/sol_execbench/test_public_contract_guardrails.py` |
| **Estimated runtime** | ~30 seconds |

---

## Sampling Rate

- **After every task commit:** Run `uv run pytest tests/sol_execbench/test_dataset_contract.py tests/sol_execbench/test_download_solexecbench.py -x`
- **After every plan wave:** Run `uv run pytest tests/sol_execbench/test_dataset_contract.py tests/sol_execbench/test_download_solexecbench.py tests/sol_execbench/test_public_contract_guardrails.py`
- **Before `$gsd-verify-work`:** Full relevant suite must be green
- **Max feedback latency:** 60 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 53-01-01 | 01 | 1 | DATA-01 | path traversal | Category validation rejects unknown categories | unit | `uv run pytest tests/sol_execbench/test_dataset_contract.py -x` | missing | pending |
| 53-01-02 | 01 | 1 | DATA-01 | path traversal | Layout diagnostics use fixed categories and shallow counts only | unit | `uv run pytest tests/sol_execbench/test_dataset_contract.py -x` | missing | pending |
| 53-01-03 | 01 | 1 | DATA-02, DATA-04 | claim overreach | Manifest is deterministic and claim boundaries stay false | unit | `uv run pytest tests/sol_execbench/test_dataset_contract.py -x` | missing | pending |
| 53-02-01 | 02 | 2 | DATA-03 | unsafe input | Downloader CLI validates categories, output root, manifest, revision, and verify-only modes | unit | `uv run pytest tests/sol_execbench/test_download_solexecbench.py -x` | missing | pending |
| 53-02-02 | 02 | 2 | DATA-02, DATA-03 | tampering | Downloader skips identical files and rejects divergent files unless `--force` | unit | `uv run pytest tests/sol_execbench/test_download_solexecbench.py -x` | missing | pending |
| 53-02-03 | 02 | 2 | DATA-03 | N/A | Aggregate shell wrapper remains valid and aligned to canonical root | syntax | `bash -n scripts/download_data.sh` | existing | pending |
| 53-03-01 | 03 | 3 | DATA-04 | claim overreach | Docs separate acquisition/layout from readiness and validation | guardrail | `uv run pytest tests/sol_execbench/test_public_contract_guardrails.py -x` | existing | pending |
| 53-03-02 | 03 | 3 | DATA-04 | claim overreach | Guardrails fail on missing non-validation wording | guardrail | `uv run pytest tests/sol_execbench/test_public_contract_guardrails.py tests/sol_execbench/test_dataset_contract.py -x` | existing | pending |
| 53-03-03 | 03 | 3 | DATA-01..DATA-04 | N/A | Phase state and traceability reflect complete DATA-01..DATA-04 only | artifact | `gsd-sdk query roadmap.analyze` | existing | pending |

---

## Wave 0 Requirements

- [ ] `tests/sol_execbench/test_dataset_contract.py` — new fixture-based tests for DATA-01, DATA-02, and DATA-04.
- [ ] `tests/sol_execbench/test_download_solexecbench.py` — new monkeypatched downloader CLI tests for DATA-03.

---

## Manual-Only Verifications

All phase behaviors have automated verification. Real public dataset download
is intentionally manual/user-run and is not required for Phase 53 tests.

---

## Validation Sign-Off

- [x] All tasks have automated verify commands or Wave 0 dependencies
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all missing references
- [x] No watch-mode flags
- [x] Feedback latency < 60s
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** approved 2026-05-23
