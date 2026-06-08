# Phase 167 Context: RDNA4 Clock-Lock Evidence

## Goal

Collect host-level clock-lock and reset evidence required to upgrade RDNA4
timing from functional/profiler evidence to benchmark-grade timing authority.

## Depends On

- Phase 166: RDNA4 rocprof timing closure

## Scope

- Record `rocm-smi` pre/run/post clock state and reset behavior.
- Document required sudoers or operator-managed host clock policy.
- Add guardrails that block authoritative timing claims when clock evidence is
  missing or unlocked.

## Primary Deliverable

- Clock-lock evidence report and host setup documentation.
