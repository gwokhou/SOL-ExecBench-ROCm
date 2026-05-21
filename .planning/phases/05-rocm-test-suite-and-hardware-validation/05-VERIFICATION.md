---
phase: 05-rocm-test-suite-and-hardware-validation
status: partial_rdna4_focused_pass_pending_cdna3
verified_at: 2026-05-21
---

# Phase 05 Verification

## Result

PARTIAL HARDWARE VALIDATION.

The local environment has been switched to PyTorch ROCm and focused Phase 5
validation passes on the visible RDNA 4 `gfx1200` GPU. The phase cannot be
marked fully complete because no CDNA 3 run was recorded and the full adapted
suite has not yet been run.

## Requirement Coverage

| Requirement | Result | Evidence |
| --- | --- | --- |
| TEST-01 | PASS | Pytest markers, example/e2e native language groups, CLI compile wording, and representative trace fixture hardware strings now use ROCm/AMD semantics. |
| TEST-02 | PASS | `requires_rocm`, `requires_rdna4`, `requires_cdna3`, unsupported AMD architecture, and unavailable ROCm skip paths are implemented in `tests/conftest.py`. |
| TEST-03 | PARTIAL | Locally feasible audit, driver, reward-hack, and example consistency tests pass under PyTorch ROCm. Full repository suite run is still pending. |
| TEST-04 | PARTIAL | PyTorch ROCm sees `AMD Radeon Graphics` / `gfx1200`; focused Phase 5 validation passed. Full adapted suite evidence is still pending. |
| TEST-05 | PENDING | No CDNA 3 full-suite run was recorded. |
| TEST-06 | PASS | Reward-hack tests remain active; focused reward-hack suite passed locally. |

## Tests

- `uv run --no-sync pytest tests/sol_execbench/test_rocm_test_suite_audit.py tests/sol_execbench/test_rocm_library_examples.py tests/examples/test_examples.py -k consistency` -> 11 passed after ROCm switch.
- `uv run --no-sync pytest tests/sol_execbench/driver/test_problem_packager.py tests/sol_execbench/core/bench/test_reward_hack.py` -> 42 passed after ROCm switch.
- `uv run --no-sync python -c "import torch; x=torch.ones(4, device='cuda'); y=x+1; torch.cuda.synchronize(); print(y.cpu().tolist())"` -> `[2.0, 2.0, 2.0, 2.0]`.
- `uv run --no-sync pytest tests/sol_execbench/driver/test_problem_packager.py tests/sol_execbench/core/bench/test_reward_hack.py tests/sol_execbench/test_e2e.py --collect-only` -> 48 collected.
- `uv run --no-sync ruff check tests/conftest.py tests/sol_execbench/test_rocm_test_suite_audit.py tests/examples/test_examples.py tests/sol_execbench/test_e2e.py src/sol_execbench/cli/main.py tests/sol_execbench/core/bench/test_reward_hack.py tests/sol_execbench/driver/test_problem_packager.py` -> passed.

## Hardware Evidence

- `rocminfo` sees AMD GPU agent `gfx1200`.
- `rocm-smi` sees one AMD GPU device.
- `uv run --no-sync python ...` sees PyTorch ROCm (`2.10.0+rocm7.1`) with
  `torch.version.hip == 7.1.25424`, `torch.version.cuda is None`, and
  `gcnArchName == gfx1200`.
- Runtime warning observed:
  `/opt/amdgpu/share/libdrm/amdgpu.ids: No such file or directory`; PyTorch
  still detects and executes on the GPU.

## Residual Risk

Phase 5 should remain open until the full adapted suite is run under PyTorch
ROCm on RDNA 4 and CDNA 3, or the project explicitly relaxes TEST-04/TEST-05.
