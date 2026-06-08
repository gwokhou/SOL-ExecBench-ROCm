# Phase 142 Context

## Goal

Provide and verify a host setup path for passwordless `rocm-smi` sudoers
coverage needed by RDNA4 clock lock and reset commands.

## Inputs

- v1.30 Phase 139 evidence:
  `.planning/milestones/v1.30-phases/139-rdna4-locked-clock-timing-and-profiler-evidence/139-SUMMARY.md`
- Clock-lock implementation:
  `src/sol_execbench/core/bench/clock_lock.py`
- Existing Docker sudoers behavior:
  `docker/Dockerfile`

## Key Context

- Phase 139 found `sudo rocm-smi --setsclk 2`,
  `sudo rocm-smi --setmclk 5`, and `sudo rocm-smi --resetclocks` still
  required a password.
- `clock_lock.py` uses `sudo -n` with the resolved `rocm-smi` path, so
  password prompts are infrastructure/setup failures rather than benchmark
  failures.
- The sudoers fix must be auditable and narrow enough to cover required clock
  commands without granting broad shell access.
- Live install may require operator approval because it writes to
  `/etc/sudoers.d`.
