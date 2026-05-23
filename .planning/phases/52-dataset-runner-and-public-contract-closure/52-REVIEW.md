---
phase: 52-dataset-runner-and-public-contract-closure
reviewed: 2026-05-23T11:30:44Z
depth: standard
files_reviewed: 8
files_reviewed_list:
  - scripts/run_dataset.py
  - src/sol_execbench/core/scoring/amd_score.py
  - tests/sol_execbench/test_run_dataset_amd_score.py
  - tests/sol_execbench/test_amd_native_score.py
  - tests/sol_execbench/test_public_contract_guardrails.py
  - tests/sol_execbench/test_v1_9_validation_closure.py
  - docs/analysis.md
  - docs/internal/solar_derivation_contract.md
findings:
  critical: 1
  warning: 1
  info: 0
  total: 2
status: issues_found
---

# Phase 52: Code Review Report

**Reviewed:** 2026-05-23T11:30:44Z
**Depth:** standard
**Files Reviewed:** 8
**Status:** issues_found

## Summary

Reviewed the Phase 52 dataset-runner closure changes against the requested AMD score, SOLAR sidecar, public-contract, and claim-boundary requirements. The pass-through sidecar/report flow is mostly scoped correctly, but sidecar filenames are built from unsanitized benchmark identifiers, which allows path traversal outside the requested sidecar directory. The new sidecar tests cover happy-path names but miss adversarial identifier cases.

## Narrative Findings (AI reviewer)

## Critical Issues

### CR-01: Sidecar filenames allow path traversal outside the requested output directory

**Classification:** BLOCKER
**File:** `scripts/run_dataset.py:510`
**Issue:** `definition.name` and `trace.workload.uuid` are interpolated directly into sidecar filenames for both `--solar-derivation` and `--amd-sol-bound-dir` (`lines 510-533`). These values come from dataset `definition.json`, `workload.jsonl`, and trace payloads, and the Pydantic `NonEmptyString` contract only enforces non-empty strings, not path-safe names. A benchmark with a name or UUID containing `/`, `..`, or path separators can make `sidecar_path` resolve outside the operator-selected sidecar directory and overwrite arbitrary files writable by the runner. This violates the sidecar-only boundary because untrusted benchmark metadata controls filesystem placement.
**Fix:**
```python
def _safe_sidecar_stem(*parts: str) -> str:
    raw = ".".join(parts)
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw)
    if safe in {"", ".", ".."} or safe.startswith("."):
        raise ValueError(f"unsafe sidecar identifier: {raw!r}")
    return safe

stem = _safe_sidecar_stem(definition.name, trace.workload.uuid)
sidecar_path = solar_derivation_dir / f"{stem}.solar-derivation.json"
resolved = sidecar_path.resolve()
if not resolved.is_relative_to(solar_derivation_dir.resolve()):
    raise ValueError(f"sidecar path escapes output directory: {sidecar_path}")
```
Apply the same helper to AMD SOL v2 sidecars, and add regression tests with names/UUIDs such as `../escape` and `nested/name`.

## Warnings

### WR-01: New sidecar tests only cover safe fixture identifiers

**Classification:** WARNING
**File:** `tests/sol_execbench/test_run_dataset_amd_score.py:125`
**Issue:** The new sidecar coverage asserts expected files only for `matmul_demo.matmul-workload...` happy-path identifiers. There is no test proving sidecar generation rejects or sanitizes path separators, dot segments, leading dots, or duplicate-normalized names. This gap is why the path traversal defect above can ship despite direct tests for sidecar generation.
**Fix:** Add tests around `build_amd_score_reports_for_problem` and the dataset runner path that use malicious or path-shaped `Definition.name` and `Workload.uuid` values, then assert the implementation either rejects them with a clear error or writes only beneath the requested sidecar directory.

---

_Reviewed: 2026-05-23T11:30:44Z_
_Reviewer: the agent (gsd-code-reviewer)_
_Depth: standard_
