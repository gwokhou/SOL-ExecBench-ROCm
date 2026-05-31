---
phase: 83-closure-contracts-and-provenance-foundation
reviewed: 2026-05-31T07:42:14Z
depth: standard
files_reviewed: 5
files_reviewed_list:
  - src/sol_execbench/core/dataset/execution_closure.py
  - scripts/run_dataset.py
  - tests/sol_execbench/test_execution_closure_contract.py
  - tests/sol_execbench/test_run_dataset_execution_closure.py
  - tests/sol_execbench/test_public_contract_guardrails.py
findings:
  critical: 1
  warning: 1
  info: 0
  total: 2
status: findings
---

# Phase 83: Code Review Report

**Reviewed:** 2026-05-31T07:42:14Z
**Depth:** standard
**Files Reviewed:** 5
**Status:** findings

## Summary

Reviewed the new execution closure contract, runner delegation, and CPU-safe guardrail tests. The focused verification command passes, but the sidecar contract still serializes local absolute provenance paths/raw argv and the new models are not strict about unknown fields, which undermines the Phase 83 contract and privacy goals.

Verification run:
`UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_execution_closure_contract.py tests/sol_execbench/test_run_dataset_execution_closure.py tests/sol_execbench/test_public_contract_guardrails.py::test_v1_11_execution_closure_fields_remain_sidecar_only -q`

Result: `13 passed in 1.38s`

## Critical Issues

### CR-01: Runner Closure Provenance Persists Absolute Local Paths And Raw CLI Args

**Classification:** BLOCKER
**File:** `scripts/run_dataset.py:1228`
**Issue:** The execution closure provenance stores `sys.argv[1:]` plus resolved absolute paths for `dataset_root`, `output_dir`, `summary_path`, ready-subset/readiness/manifest paths, config path, and derived evidence paths. These values are included in the new checksum-backed sidecar. That leaks local workspace/user paths when reports are shared and makes otherwise identical closure reports/checksums differ across machines or checkout locations. This violates the Phase 83 provenance policy to prefer relative refs/checksums and keep sensitive/noisy data out of sidecars.
**Fix:** Normalize provenance before report construction. Store bounded refs relative to `output_dir`, `problems_dir`, or repo root where possible; avoid raw argv or redact path-valued args.

```python
provenance = {
    "command_args": _normalized_command_args(args),
    "dataset_root": _relative_ref(problems_dir, ROOT),
    "output_dir": _relative_ref(output_dir, ROOT),
    "summary_path": _relative_ref(output_dir / "summary.json", output_dir),
    "ready_subset_path": _relative_ref(args.ready_subset.resolve(), ROOT) if args.ready_subset else None,
    "readiness_path": _relative_ref(args.readiness.resolve(), ROOT) if args.readiness else None,
    "dataset_manifest_path": _relative_ref(args.dataset_manifest.resolve(), ROOT) if args.dataset_manifest else None,
    "config_path": _relative_ref(config_path, output_dir) if config_path else None,
    "derived_evidence": {
        "amd_score_report": _relative_ref(args.amd_score_report.resolve(), output_dir) if args.amd_score_report else None,
        "amd_sol_bound_dir": _relative_ref(args.amd_sol_bound_dir.resolve(), output_dir) if args.amd_sol_bound_dir else None,
        "solar_derivation": _relative_ref(args.solar_derivation.resolve(), output_dir) if args.solar_derivation else None,
        "timing_evidence_dir": _relative_ref(args.timing_evidence_dir.resolve(), output_dir) if args.timing_evidence_dir else None,
    },
}
```

Add a runner test that invokes the script with absolute `tmp_path` arguments and asserts no serialized provenance string contains `str(tmp_path)`.

## Warnings

### WR-01: Execution Closure Models Are Not Strict About Unknown Fields

**Classification:** WARNING
**File:** `src/sol_execbench/core/dataset/execution_closure.py:47`
**Issue:** The new sidecar is described as a strict contract, but most Pydantic models use the default extra-field behavior, which silently ignores unknown keys, while `ExecutionClosureProvenance` explicitly uses `extra="allow"`. That lets typos or future/raw provenance fields pass validation without contract review, and it creates a path for unbounded payloads to enter the sidecar provenance. The tests currently assert vocabulary and serialization, but they do not verify that unexpected fields are rejected.
**Fix:** Set `ConfigDict(extra="forbid")` on the closure models, then add tests that unknown record/provenance/report fields raise `ValidationError`.

```python
class ExecutionClosureRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ...

class ExecutionClosureProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ...

class ExecutionClosureReport(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ...
```

---

_Reviewed: 2026-05-31T07:42:14Z_
_Reviewer: the agent (gsd-code-reviewer)_
_Depth: standard_
