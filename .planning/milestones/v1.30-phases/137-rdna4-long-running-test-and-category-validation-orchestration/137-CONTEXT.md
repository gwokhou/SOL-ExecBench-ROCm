# Phase 137: RDNA4 Long-Running Test and Category Validation Orchestration - Context

## Scope

Phase 137 turns the Phase 136 validation scope into a resumable RDNA4
orchestration path for adapted pytest and category validation. It covers
requirements RDNA4-VAL-02, RDNA4-VAL-03, RDNA4-CAT-01, RDNA4-CAT-02,
RDNA4-CAT-03, and RDNA4-CAT-04.

This phase may run real RDNA4 tests when `gfx1200` hardware is visible. It does
not execute the full dataset denominator; Phase 138 owns the full dataset run.
It does not collect benchmark-grade timing; Phase 139 owns clock-lock,
`rocprofv3`, and timing stability evidence.

## Observed Local Hardware

Local preflight on 2026-06-07 observed:

- `rocminfo`: HSA runtime 1.18, GPU agent `gfx1200`, marketing name
  `AMD Radeon Graphics`.
- `rocm-smi`: one AMD GPU visible, low-power warning, auto performance state,
  SCLK shown as 0 MHz while idle.
- `lspci`: Navi 44 `[Radeon RX 9060 XT]` plus unrelated NVIDIA GB206 device.

The NVIDIA device is not validation evidence for this milestone.

## Existing Assets

- Hardware marker behavior: `tests/conftest.py` supports `requires_rdna4`.
- RDNA4 docs and historical evidence:
  - `docs/internal/rdna4_v1_9_validation_evidence.md`
  - `.planning/milestones/v1.0-phases/05-rocm-test-suite-and-hardware-validation/`
  - `docs/rocm.md`
- Category/example coverage:
  - `tests/examples/test_examples.py`
  - `tests/examples/test_rocm_cli_paths.py`
  - `tests/sol_execbench/test_rocm_library_examples.py`
  - `tests/sol_execbench/test_rocm_library_readiness_docs.py`
  - `tests/sol_execbench/test_rocm_diagnostics_reporting.py`
- Bounded validation helper:
  - `scripts/release_candidate_validation.py`

## Long-Running Policy

Jobs may run for many hours. A healthy process should be polled and allowed to
continue. Do not terminate a process solely because it is long-running. Preserve
stdout/stderr logs, command lines, environment preflight output, and partial
pytest reports. Use resumable reruns and existing artifacts where supported.

## Claim Boundaries

- Passing RDNA4 marker/category tests is not full dataset validation.
- Category readiness is not benchmark-grade validation unless Phase 138-141
  evidence exists.
- Docker/container evidence remains distinct from native-host validation.
- CDNA3/MI300X and CDNA4 claims are outside this phase.
