---
phase: 05-rocm-test-suite-and-hardware-validation
status: completed_with_cdna3_deferred
verified_at: 2026-05-21
---

# Phase 05 Verification

## Result

PASSED FOR CURRENT MILESTONE; CDNA3 DEFERRED.

The local environment has been switched to PyTorch ROCm and the full adapted
test suite passes on the visible RDNA 4 `gfx1200` GPU. No CDNA 3 run was
recorded on this machine, so TEST-05 is explicitly deferred to a later
milestone by project decision on 2026-05-21.

## Requirement Coverage

| Requirement | Result | Evidence |
| --- | --- | --- |
| TEST-01 | PASS | Pytest markers, example/e2e native language groups, CLI compile wording, and representative trace fixture hardware strings now use ROCm/AMD semantics. |
| TEST-02 | PASS | `requires_rocm`, `requires_rdna4`, `requires_cdna3`, unsupported AMD architecture, and unavailable ROCm skip paths are implemented in `tests/conftest.py`. |
| TEST-03 | PASS | Full local adapted suite passed under PyTorch ROCm on RDNA 4: `462 passed, 58 skipped`. |
| TEST-04 | PASS | PyTorch ROCm sees `AMD Radeon Graphics` / `gfx1200`; full local adapted suite passed. |
| TEST-05 | DEFERRED | No CDNA 3 full-suite run was recorded; deferred to a later milestone by project decision on 2026-05-21. |
| TEST-06 | PASS | Reward-hack tests remain active; focused reward-hack suite passed locally. |

## Tests

- `uv run --no-sync pytest tests/sol_execbench/test_rocm_test_suite_audit.py tests/sol_execbench/test_rocm_library_examples.py tests/examples/test_examples.py -k consistency` -> 11 passed after ROCm switch.
- `uv run --no-sync pytest tests/sol_execbench/driver/test_problem_packager.py tests/sol_execbench/core/bench/test_reward_hack.py` -> 42 passed after ROCm switch.
- `uv run --no-sync python -c "import torch; x=torch.ones(4, device='cuda'); y=x+1; torch.cuda.synchronize(); print(y.cpu().tolist())"` -> `[2.0, 2.0, 2.0, 2.0]`.
- `uv run --no-sync pytest tests/sol_execbench/driver/test_problem_packager.py tests/sol_execbench/core/bench/test_reward_hack.py tests/sol_execbench/test_e2e.py --collect-only` -> 48 collected.
- `uv run --no-sync ruff check tests/conftest.py tests/sol_execbench/test_rocm_test_suite_audit.py tests/examples/test_examples.py tests/sol_execbench/test_e2e.py src/sol_execbench/cli/main.py tests/sol_execbench/core/bench/test_reward_hack.py tests/sol_execbench/driver/test_problem_packager.py` -> passed.
- `uv run --no-sync pytest tests/examples/test_examples.py::test_example -k "rmsnorm_hip or flux_rope_hip"` -> 2 passed.
- `uv run --no-sync pytest tests/sol_execbench/test_e2e.py::test_cli_gqa_paged_decode` -> 1 skipped because external safetensors inputs are not available locally.
- `uv run --no-sync pytest tests/` -> 462 passed, 58 skipped on RDNA 4 `gfx1200`.

## Hardware Evidence

- `rocminfo` sees AMD GPU agent `gfx1200`.
- `rocm-smi` sees one AMD GPU device.
- `uv run --no-sync python ...` sees PyTorch ROCm (`2.10.0+rocm7.1`) with
  `torch.version.hip == 7.1.25424`, `torch.version.cuda is None`, and
  `gcnArchName == gfx1200`.
- `/opt/amdgpu/share/libdrm/amdgpu.ids` exists as a symlink to
  `/usr/share/libdrm/amdgpu.ids`; the earlier runtime warning is resolved.

## Deferred Follow-up

Run the full adapted suite under PyTorch ROCm on CDNA 3 (`gfx94*`) before
claiming CDNA 3 hardware support. This is no longer blocking Phase 6 in the
current milestone.
