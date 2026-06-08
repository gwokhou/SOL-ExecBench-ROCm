---
phase: 142
title: RDNA4 clock sudoers coverage
status: verified
verified_at: 2026-06-08
---

# Phase 142 Verification

## Commands

- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_setup_rocm_smi_sudoers.py tests/sol_execbench/core/bench/test_clock_lock.py tests/sol_execbench/test_run_docker_matrix_script.py -q`
- `UV_CACHE_DIR=/tmp/uv-cache uv run --with ruff ruff check scripts/setup_rocm_smi_sudoers.py tests/sol_execbench/test_setup_rocm_smi_sudoers.py`
- `UV_CACHE_DIR=/home/guohao/.cache/uv /home/guohao/.cargo/bin/uv run python scripts/setup_rocm_smi_sudoers.py --mode check --rocm-smi /usr/bin/rocm-smi --json`

## Results

- Focused pytest passed: 57 passed.
- Ruff passed for the new sudoers setup script and tests.
- Live host sudoers check passed with `status: covered` for all required
  `/usr/bin/rocm-smi` clock query, lock, and reset commands.

## Residual Risk

- `rocm-smi` still reports low-power-state warnings while idle. Phase 143 must
  determine whether lock verification succeeds under actual clock-lock
  conditions.
- The sudoers rule is path-specific. If a future host uses a different
  `rocm-smi` path, rerun the installer with `--rocm-smi <path>`.
