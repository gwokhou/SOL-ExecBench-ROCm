---
phase: 142
title: RDNA4 clock sudoers coverage
status: completed
completed_at: 2026-06-08
---

# Phase 142 Summary

## Completed

- Added `scripts/setup_rocm_smi_sudoers.py` with `print`, `check`, and
  root-only `install` modes for narrow passwordless `rocm-smi` clock command
  coverage.
- Fixed sudo install user detection so root execution defaults to `SUDO_USER`
  rather than writing rules for `root`.
- Added CPU-safe tests for sudoers content, check classification, install
  safety, and sudo-user detection.
- Extended Docker sudoers setup with command-specific `rocm-smi` clock query,
  lock, and reset coverage.
- Verified live host coverage outside the sandbox for `/usr/bin/rocm-smi`.

## Live Coverage

The post-install host check returned `status: covered` for:

- `sudo -n /usr/bin/rocm-smi --showclocks`
- `sudo -n /usr/bin/rocm-smi -s`
- `sudo -n /usr/bin/rocm-smi --showperflevel`
- `sudo -n /usr/bin/rocm-smi --showclkfrq`
- `sudo -n /usr/bin/rocm-smi --setperflevel manual`
- `sudo -n /usr/bin/rocm-smi --setperflevel auto`
- `sudo -n /usr/bin/rocm-smi --setsclk 0`
- `sudo -n /usr/bin/rocm-smi --setmclk 0`
- `sudo -n /usr/bin/rocm-smi --resetclocks`

## Boundaries

- Phase 142 proves sudoers command coverage only. It does not claim RDNA4
  clocks are locked for benchmark-grade timing.
- Phase 143 must rerun clock-lock evidence and profiler-backed timing checks
  using the now-covered commands.
