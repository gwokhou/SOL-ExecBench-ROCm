# Phase 163 Context: RDNA4 Denominator Policy Hardening

## Goal

Convert Phase 162 `reference_oom_blocked` accounting into a stable RDNA4
validation denominator policy for the current `gfx1200` 16GB device class.

## Depends On

- Phase 162: RDNA4 memory-footprint denominator policy

## Scope

- Define included, excluded, blocked, and claim-safe statuses for the 235
  problem denominator.
- Preserve the boundary that current-device OOM blockers are accounted but do
  not count as profiler-backed timing or full validation pass evidence.
- Document how `gfx1200` 16GB validation differs from larger-memory AMD
  validation.

## Primary Deliverable

- `RDNA4-DENOMINATOR-POLICY.md`
