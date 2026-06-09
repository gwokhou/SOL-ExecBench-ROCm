---
status: complete
completed_at: "2026-06-09"
---

# Quick Task Summary: Align amd-smi Clock Locking

## Result

Clock-lock support is now consistently aligned around the confirmed
`amd-smi set -l STABLE_PEAK` path.

## Changes

- Docker sudoers setup now requires `amd-smi` and grants only `amd-smi version`,
  `amd-smi set -l STABLE_PEAK`, and `amd-smi set -l AUTO`.
- `scripts/rdna4_clock_lock_workload_test.py` now uses the production
  `clock_lock.lock_clocks()` and `unlock_clocks()` helpers instead of repeating
  rejected `rocm-smi --setsclk/--setmclk` logic.
- Sudoers helper and tests are now named and scoped around
  `setup_rocm_clock_sudoers.py` and amd-smi commands.
- Public docs and codebase maps no longer describe SCLK/MCLK override
  environment variables as active clock-lock controls.

## Verification

- `uv run pytest tests/sol_execbench/core/bench/test_clock_lock.py tests/sol_execbench/test_setup_rocm_clock_sudoers.py -q`
- `uv run --with ruff ruff check scripts/setup_rocm_clock_sudoers.py scripts/rdna4_clock_lock_workload_test.py tests/sol_execbench/test_setup_rocm_clock_sudoers.py`
- `bash -n docker/entrypoint.sh`
- `git diff --check`
