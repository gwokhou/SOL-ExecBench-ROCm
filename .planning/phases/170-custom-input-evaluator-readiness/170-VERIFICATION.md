---
status: passed
phase: 170-custom-input-evaluator-readiness
verified: 2026-06-09
verifier: orchestrator-inline
---

# Phase 170 Verification

## Phase Goal

Implement deterministic benchmark-defined custom input generation so the 55 L1/L2 custom-input readiness blockers can be attempted safely on ROCm.

## Requirement Traceability

| ID | Description | Status | Evidence |
|----|-------------|--------|----------|
| CUST-01 | Evaluator can execute benchmark-defined custom input generation | Passed | `gen_custom_inputs()` in `io.py`, driver integration in `eval_driver.py` |
| CUST-02 | Evaluator validates generated custom inputs against expected schema | Passed | `_validate_custom_tensors()` checks keys, kind, dtype, shape, device |
| CUST-03 | Custom input generation is deterministic per workload | Passed | `derive_custom_input_seed()` + `isolated_torch_rng()` with CPU/CUDA state restore |
| CUST-04 | Custom input failures classified separately as gen_inputs_* | Passed | Five constants: `gen_inputs_error`, `gen_inputs_oom_blocked`, `gen_inputs_timeout`, `gen_inputs_schema_mismatch`, `gen_inputs_device_mismatch` |

## Must-Have Verification

### Truths

1. **Custom workloads generate benchmark-defined inputs before reference/candidate execution** -- VERIFIED: `gen_custom_inputs()` resolves entrypoint from staged reference, calls with `axes_and_scalars`, returns validated inputs.
2. **RNG state restored after each generation attempt** -- VERIFIED: `isolated_torch_rng()` context manager snapshots CPU and CUDA RNG state, restores in finally block.
3. **Generated inputs validated for keys, kind, dtype, shape, device** -- VERIFIED: `_validate_custom_tensors()` has branches for missing keys, unexpected keys, scalar/tensor kind mismatch, dtype mismatch, shape mismatch, device mismatch.
4. **Five gen_inputs_* failure classes** -- VERIFIED: Constants in `io.py`, diagnostic constants in `diagnostics.py`, exception classification in `_classify_custom_generation_exception()`.
5. **Supported custom-input workloads no longer blanket custom_input_blocked** -- VERIFIED: `readiness.py` lines 446-471 classify supported workloads as `ready_to_generate`, malformed as `custom_input_requires_evaluator_support`.

### Artifacts

1. `src/sol_execbench/core/bench/io.py` -- Contains `CustomInputProvenance`, `CustomInputGenerationError`, `derive_custom_input_seed()`, `isolated_torch_rng()`, `_validate_custom_tensors()`, `gen_custom_inputs()`.
2. `src/sol_execbench/driver/templates/eval_driver.py` -- Emits custom input generation diagnostics before reference/candidate execution.
3. `src/sol_execbench/core/dataset/readiness.py` -- Distinguishes supported custom inputs (ready_to_generate) from malformed (custom_input_requires_evaluator_support).
4. CPU-safe tests cover success, determinism, validation failures, driver integration, and readiness classification.

### Key Links

1. `Definition.custom_inputs_entrypoint` resolved from staged reference code and passed into input assembly -- VERIFIED in `gen_custom_inputs()` signature.
2. Workload axes plus scalar inputs used as `axes_and_scalars` -- VERIFIED in `gen_custom_inputs()` body.
3. Canonical trace JSON validity preserved -- VERIFIED: diagnostics use standard `RUNTIME_ERROR` status with `gen_inputs_*` in log text, no new trace fields.

## Test Results

| Suite | Passed | Failed | Skipped | Exit |
|-------|--------|--------|---------|------|
| tests/sol_execbench/core/bench/test_io.py | 131 | 0 | 1 | 0 |
| tests/sol_execbench/driver/test_eval_driver.py | 26 | 0 | 1 | 0 |
| tests/sol_execbench/test_dataset_inventory_readiness.py | 20 | 0 | 0 | 0 |
| tests/sol_execbench/test_e2e.py + closure tests | 49 | 0 | 0 | 0 |
| **Total** | **226** | **0** | **2** | **0** |

## Dataset Integrity

- No files under `data/` modified.
- No migrated `definition.json` or `workload.jsonl` files modified.
- 235-problem denominator stable.

## Verdict

**PASSED** -- All four requirements verified, all must-have truths confirmed in code, all test suites pass, dataset integrity preserved. Phase 170 does not claim RDNA4 coverage movement; that remains Phase 171's responsibility.
