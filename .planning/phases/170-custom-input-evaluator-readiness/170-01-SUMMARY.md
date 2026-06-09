---
phase: 170-custom-input-evaluator-readiness
plan: "01"
subsystem: evaluator
tags: [custom-inputs, readiness, validation, diagnostics, rocm, pytorch]

# Dependency graph
requires:
  - phase: 133
    provides: ROCm readiness classification infrastructure
provides:
  - Deterministic benchmark-defined custom input generation with seed/provenance tracking
  - Structured custom input validation (keys, kind, dtype, shape, device)
  - Five gen_inputs_* diagnostic failure classes
  - Precise readiness reclassification removing blanket custom_input_blocked
affects: [171, dataset-runner, coverage-recompute]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Per-workload seed derivation from definition name + workload UUID/row index"
    - "PyTorch RNG state snapshot/restore around custom input generation"
    - "Schema-validated input generation with provenance result objects"
    - "Separate gen_inputs_* failure classes distinct from reference/candidate failures"

key-files:
  created: []
  modified:
    - src/sol_execbench/core/bench/io.py
    - src/sol_execbench/driver/templates/eval_driver.py
    - src/sol_execbench/core/dataset/readiness.py
    - src/sol_execbench/core/diagnostics.py
    - tests/sol_execbench/core/bench/test_io.py
    - tests/sol_execbench/driver/test_eval_driver.py
    - tests/sol_execbench/test_dataset_inventory_readiness.py
    - tests/sol_execbench/test_e2e.py
    - tests/sol_execbench/test_execution_closure_contract.py
    - tests/sol_execbench/test_run_dataset_execution_closure.py

key-decisions:
  - "Per-workload seed derives from definition.name + workload.uuid (or row-index fallback) for deterministic reproducibility"
  - "Five separate gen_inputs_* failure classes keep input-generation errors distinct from reference/candidate runtime errors"
  - "Readiness reclassification only removes custom_input_blocked when entrypoint exists and all workloads are schema-parsed; malformed entries stay blocked"

patterns-established:
  - "CustomInputResult provenance object: entrypoint, seed, workload_uuid, row_index, generated_keys, failure_class"
  - "RNG isolation: torch.manual_seed(seed) + snapshot/restore of CPU and CUDA RNG state around generation"
  - "Validation rejects: missing keys, unexpected keys, tensor/scalar kind mismatch, dtype mismatch, shape mismatch, device mismatch"

requirements-completed: [CUST-01, CUST-02, CUST-03, CUST-04]

# Metrics
duration: pre-committed
completed: 2026-06-09
---

# Phase 170: Custom Input Evaluator Readiness Summary

**Deterministic custom input generation with schema validation, five failure-class diagnostics, and precise readiness reclassification for benchmark-defined custom inputs**

## Performance

- **Duration:** pre-committed (4 atomic commits)
- **Started:** 2026-06-09
- **Completed:** 2026-06-09
- **Tasks:** 4
- **Files modified:** 10

## Accomplishments
- Custom input assembly generates deterministic inputs with per-workload seed derivation and PyTorch RNG state isolation
- Structured validation rejects malformed generated inputs (keys, kind, dtype, shape, device) before reference/candidate execution
- Five distinct gen_inputs_* diagnostic classes separate input-generation failures from reference and candidate failures
- Readiness reclassification promotes supported custom-input workloads to ready while retaining precise blockers for malformed entries
- CPU-safe test coverage across all four surfaces (io, driver, readiness, closure)

## Task Commits

Each task was committed atomically:

1. **Task 1: Add validated deterministic custom input assembly** - `203b515` (feat)
2. **Task 2: Wire custom input diagnostics into staged evaluation** - `bd06769` (feat)
3. **Task 3: Reclassify supported custom input readiness without coverage recompute** - `ca63cdf` (feat)
4. **Task 4: Run focused integration and guardrail verification** - `e365e13` (test)

## Files Created/Modified
- `src/sol_execbench/core/bench/io.py` - Custom input generation, seed/provenance, RNG isolation, validation helpers (+285 lines)
- `src/sol_execbench/core/diagnostics.py` - Five gen_inputs_* failure-class constants
- `src/sol_execbench/driver/templates/eval_driver.py` - Custom input generation before reference/candidate execution, diagnostic emission
- `src/sol_execbench/core/dataset/readiness.py` - Precise custom-input readiness decision replacing blanket blocker (+238/-84)
- `tests/sol_execbench/core/bench/test_io.py` - Determinism, RNG state, validation failure tests (+129 lines)
- `tests/sol_execbench/driver/test_eval_driver.py` - CPU-safe subprocess tests for success, schema mismatch, non-OOM errors (+136 lines)
- `tests/sol_execbench/test_dataset_inventory_readiness.py` - Supported and malformed custom-input readiness fixtures (+201 lines)
- `tests/sol_execbench/test_e2e.py` - Custom input example representation in CPU-safe coverage
- `tests/sol_execbench/test_execution_closure_contract.py` - gen_inputs_* class preservation assertion
- `tests/sol_execbench/test_run_dataset_execution_closure.py` - Closure guardrail assertions for gen_inputs_* classes

## Decisions Made
- Per-workload seed derives from `definition.name` + `workload.uuid` with row-index fallback for deterministic reproducibility across runs
- Five separate `gen_inputs_*` failure classes (`error`, `oom_blocked`, `timeout`, `schema_mismatch`, `device_mismatch`) keep input-generation errors distinct from reference/candidate runtime errors
- Readiness reclassification only removes `custom_input_blocked` when `custom_inputs_entrypoint` exists and all workloads are structurally supported; malformed entries remain explicit blockers

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Phase 171 can consume the custom input generation and readiness reclassification to recompute RDNA4 coverage with the 55 custom-input problems now potentially unblocked
- No dataset files were modified; coverage recompute remains Phase 171's responsibility
- All existing readiness blockers for non-custom-input categories (safetensors, Quant, FlashInfer, NVIDIA) remain unchanged

---
*Phase: 170-custom-input-evaluator-readiness*
*Completed: 2026-06-09*
