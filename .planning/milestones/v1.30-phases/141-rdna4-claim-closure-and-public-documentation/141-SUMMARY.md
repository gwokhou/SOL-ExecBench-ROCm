---
phase: 141
title: RDNA4 claim closure and public documentation
status: completed
completed_at: 2026-06-08
---

# Phase 141 Summary

## Completed

- Updated README validation status with the bounded RDNA4 `gfx1200` ready-subset
  result, derived score counts, sidecar coverage counts, timing boundary, and
  isolated-run rule for OOM-heavy jobs.
- Updated `docs/CLAIMS.md`, `docs/research_preview.md`,
  `docs/release_candidate_validation.md`, and `docs/rocm.md` so RDNA4 wording
  matches Phase 138-140 evidence without upgrading unrelated claims.
- Added CPU-safe public-contract guardrail coverage for RDNA4 bounded evidence
  counts, non-authoritative timing, isolated long-job execution, and forbidden
  claim upgrades.

## Final Public Boundary

The public RDNA4 claim is a bounded ready-subset `gfx1200` evidence result:
121 ready problems, 1907 attempted workloads, 1761 passed workloads, 146 failed
workloads, 86 OK problems, 35 FAIL problems, and 12 explicit `missing_trace`
workload records.

Derived evidence is also bounded: 1895 score records, 172 scored, 1723 unscored,
and 1839 AMD SOL/SOLAR sidecar pairs after 56 temporary long-tail sidecar
exclusions.

## Boundaries

- Timing remains non-authoritative until clock-lock/reset sudoers coverage and
  profiler-backed timing evidence are rerun.
- RDNA4 evidence does not claim full 235-problem paper validation, upstream
  SOLAR parity, NVIDIA B200 equivalence, hosted leaderboard authority,
  CDNA3/MI300X validation, or CDNA4 validation.
- Long memory-heavy RDNA4 derived/dataset work should run in isolated
  `systemd-run --user` units through `scripts/run_derived_isolated.py`, with
  Codex polling logs/status instead of owning heavy child processes.

