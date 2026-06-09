---
status: in_progress
created_at: "2026-06-09T04:31:25.868Z"
---

# Quick Task Plan: Align amd-smi Clock Locking

## Goal

Align clock-lock support around the confirmed `amd-smi set -l STABLE_PEAK`
implementation path.

## Steps

- Update Docker sudoers setup to grant the `amd-smi` commands used by runtime.
- Replace the RDNA4 workload evidence script's duplicated `rocm-smi` DPM lock
  logic with the shared `clock_lock` helper.
- Rename and update sudoers tests to match the `amd-smi` helper.
- Update docs and claims that still describe the old SCLK/MCLK override path.
- Run focused tests and syntax checks.
