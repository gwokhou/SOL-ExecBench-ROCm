---
phase: 05-rocm-test-suite-and-hardware-validation
status: blocked_pending_hardware
verified_at: 2026-05-21
---

# Phase 05 Verification

## Result

BLOCKED PENDING HARDWARE VALIDATION.

The ROCm test-suite migration that can be verified locally passed. The phase
cannot be marked fully complete because the current Python environment is not
PyTorch ROCm and no full adapted-suite run on CDNA 3 was recorded.

## Requirement Coverage

| Requirement | Result | Evidence |
| --- | --- | --- |
| TEST-01 | PASS | Pytest markers, example/e2e native language groups, CLI compile wording, and representative trace fixture hardware strings now use ROCm/AMD semantics. |
| TEST-02 | PASS | `requires_rocm`, `requires_rdna4`, `requires_cdna3`, unsupported AMD architecture, and unavailable ROCm skip paths are implemented in `tests/conftest.py`. |
| TEST-03 | PARTIAL | Locally feasible audit, driver, reward-hack, and example consistency tests pass. Full ROCm environment run is pending because active PyTorch is `2.10.0+cu130`, not ROCm. |
| TEST-04 | PENDING | `rocminfo` sees `gfx1200`, but full RDNA 4 suite evidence requires PyTorch ROCm visibility and was not collected. |
| TEST-05 | PENDING | No CDNA 3 full-suite run was recorded. |
| TEST-06 | PASS | Reward-hack tests remain active; focused reward-hack suite passed locally. |

## Tests

- `uv run --no-sync pytest tests/sol_execbench/test_rocm_test_suite_audit.py tests/sol_execbench/test_rocm_library_examples.py tests/examples/test_examples.py -k consistency` -> 11 passed.
- `uv run --no-sync pytest tests/sol_execbench/driver/test_problem_packager.py tests/sol_execbench/core/bench/test_reward_hack.py` -> 42 passed.
- `uv run --no-sync pytest tests/sol_execbench/driver/test_problem_packager.py tests/sol_execbench/core/bench/test_reward_hack.py tests/sol_execbench/test_e2e.py --collect-only` -> 48 collected.
- `uv run --no-sync ruff check tests/conftest.py tests/sol_execbench/test_rocm_test_suite_audit.py tests/examples/test_examples.py tests/sol_execbench/test_e2e.py src/sol_execbench/cli/main.py tests/sol_execbench/core/bench/test_reward_hack.py tests/sol_execbench/driver/test_problem_packager.py` -> passed.

## Hardware Evidence

- `rocminfo` sees AMD GPU agent `gfx1200`.
- `rocm-smi` sees one AMD GPU device.
- `uv run --no-sync python ...` sees CUDA PyTorch (`2.10.0+cu130`) with
  `torch.version.hip is None`; this is not a valid PyTorch ROCm suite run.

## Residual Risk

Phase 5 should remain open until the adapted suite is run under PyTorch ROCm on
RDNA 4 and CDNA 3, or the project explicitly relaxes TEST-04/TEST-05.

