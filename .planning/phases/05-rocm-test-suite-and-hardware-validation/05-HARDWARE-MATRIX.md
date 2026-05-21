# Phase 05 Hardware Matrix

## Scope

This file records hardware validation evidence for the adapted ROCm test suite.
Do not mark a hardware class as passed unless the suite was run on that class of
AMD GPU in a ROCm >= 7.0 environment.

## Local Environment

Initial local tool visibility:

- `rocminfo`: present at `/usr/bin/rocminfo`
- `rocm-smi`: present at `/usr/bin/rocm-smi`
- `rocminfo`: AMD GPU agent `gfx1200` visible (`AMD Radeon Graphics`)
- `rocm-smi`: one AMD GPU device visible, device ID `0x7590`
- PyTorch environment under `uv run --no-sync`: `torch 2.10.0+rocm7.1`,
  `torch.version.hip == 7.1.25424`, `torch.version.cuda is None`,
  `torch.cuda.is_available() is True`, one device visible:
  `AMD Radeon Graphics` / `gfx1200`.
- ROCm tensor smoke test passed:
  `torch.ones(4, device="cuda") + 1` synchronized and returned
  `[2.0, 2.0, 2.0, 2.0]`.
- `/opt/amdgpu/share/libdrm/amdgpu.ids` is present as a symlink to
  `/usr/share/libdrm/amdgpu.ids`; the earlier runtime warning is resolved.

## Hardware Targets

| Target | Architecture Match | Suite Status | Evidence |
|--------|--------------------|--------------|----------|
| RDNA 4 | `gfx12*` such as `gfx1200` | Full local pass | PyTorch ROCm sees `AMD Radeon Graphics` / `gfx1200`; full local adapted suite passed with `462 passed, 58 skipped`. |
| CDNA 3 | `gfx94*` such as `gfx942` | Pending | No full adapted suite run recorded yet in this phase. |

## Local Test Commands

- `uv run --no-sync pytest tests/sol_execbench/test_rocm_test_suite_audit.py tests/sol_execbench/test_rocm_library_examples.py tests/examples/test_examples.py -k consistency`
  - Result after ROCm switch: 11 passed
- `uv run --no-sync pytest tests/sol_execbench/driver/test_problem_packager.py tests/sol_execbench/core/bench/test_reward_hack.py`
  - Result after ROCm switch: 42 passed
- `uv run --no-sync python -c "import torch; x=torch.ones(4, device='cuda'); y=x+1; torch.cuda.synchronize(); print(y.cpu().tolist())"`
  - Result after ROCm switch: `[2.0, 2.0, 2.0, 2.0]`
- `uv run --no-sync pytest tests/sol_execbench/driver/test_problem_packager.py tests/sol_execbench/core/bench/test_reward_hack.py tests/sol_execbench/test_e2e.py --collect-only`
  - Result: 48 tests collected
- `uv run --no-sync ruff check tests/conftest.py tests/sol_execbench/test_rocm_test_suite_audit.py tests/examples/test_examples.py tests/sol_execbench/test_e2e.py src/sol_execbench/cli/main.py tests/sol_execbench/core/bench/test_reward_hack.py tests/sol_execbench/driver/test_problem_packager.py`
  - Result: all checks passed
- `uv run --no-sync pytest tests/examples/test_examples.py::test_example -k "rmsnorm_hip or flux_rope_hip"`
  - Result after HIP native example fixes: 2 passed
- `uv run --no-sync pytest tests/sol_execbench/test_e2e.py::test_cli_gqa_paged_decode`
  - Result: 1 skipped because external safetensors inputs are not available locally
- `uv run --no-sync pytest tests/`
  - Result on RDNA 4 `gfx1200`: 462 passed, 58 skipped

## Hardware Evidence Required Before Completion Claims

- RDNA 4 pass evidence has been collected on local `gfx1200`.
- CDNA 3 pass evidence requires running the adapted suite in a PyTorch ROCm
  environment on `gfx94*`.
- The current local machine now has PyTorch ROCm visibility for `gfx1200`.
- TEST-05 remains pending until the adapted suite runs on CDNA 3.
