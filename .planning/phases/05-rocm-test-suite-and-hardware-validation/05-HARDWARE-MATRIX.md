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
- PyTorch environment under `uv run --no-sync`: `torch 2.10.0+cu130`,
  `torch.version.hip is None`, `torch.cuda.is_available() is True` for an
  NVIDIA GeForce RTX 5060 Ti. This is not a PyTorch ROCm validation
  environment.

## Hardware Targets

| Target | Architecture Match | Suite Status | Evidence |
|--------|--------------------|--------------|----------|
| RDNA 4 | `gfx12*` such as `gfx1200` | Pending | `rocminfo` sees `gfx1200`, but the active PyTorch environment is CUDA (`+cu130`) rather than ROCm, so the full adapted suite was not run as ROCm evidence. |
| CDNA 3 | `gfx94*` such as `gfx942` | Pending | No full adapted suite run recorded yet in this phase. |

## Local Test Commands

- `uv run --no-sync pytest tests/sol_execbench/test_rocm_test_suite_audit.py tests/sol_execbench/test_rocm_library_examples.py tests/examples/test_examples.py -k consistency`
  - Result: 11 passed
- `uv run --no-sync pytest tests/sol_execbench/driver/test_problem_packager.py tests/sol_execbench/core/bench/test_reward_hack.py`
  - Result: 42 passed
- `uv run --no-sync pytest tests/sol_execbench/driver/test_problem_packager.py tests/sol_execbench/core/bench/test_reward_hack.py tests/sol_execbench/test_e2e.py --collect-only`
  - Result: 48 tests collected
- `uv run --no-sync ruff check tests/conftest.py tests/sol_execbench/test_rocm_test_suite_audit.py tests/examples/test_examples.py tests/sol_execbench/test_e2e.py src/sol_execbench/cli/main.py tests/sol_execbench/core/bench/test_reward_hack.py tests/sol_execbench/driver/test_problem_packager.py`
  - Result: all checks passed

## Hardware Evidence Required Before Completion Claims

- RDNA 4 pass evidence requires running the adapted suite in a PyTorch ROCm
  environment on `gfx12*`.
- CDNA 3 pass evidence requires running the adapted suite in a PyTorch ROCm
  environment on `gfx94*`.
- The current local machine has ROCm runtime visibility for `gfx1200`, but the
  active Python environment is not ROCm-enabled, so TEST-04 and TEST-05 remain
  pending.
