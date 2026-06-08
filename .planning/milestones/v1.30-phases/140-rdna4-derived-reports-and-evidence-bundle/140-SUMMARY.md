---
phase: 140
title: RDNA4 derived reports and evidence bundle
status: completed
completed_at: 2026-06-08
---

# Phase 140 Summary

## Completed

- Generated RDNA4 AMD-native score report at
  `out/rdna4-derived-reports/amd-score.json`.
- Generated 1839 AMD SOL v2 sidecars and 1839 SOLAR derivation sidecars under
  `out/rdna4-derived-reports/`.
- Generated paper denominator, parity gap, AMD bound sanity, consistency,
  claim-upgrade, and trust-summary reports under
  `out/rdna4-derived-reports/reports/`.
- Generated bundle manifest at
  `out/rdna4-derived-reports/bundle/evidence-bundle.json`.
- Added `scripts/run_derived_isolated.py` for resumable per-problem derived
  jobs with optional `systemd-run --user` memory/swap isolation.

## Evidence

- Score report: 1895 score records, 172 scored, 1723 unscored.
- Sidecar audit: 1839 traces with both AMD SOL and SOLAR sidecars, 56
  temporarily excluded sidecar workloads, 0 unexcluded missing sidecars.
- Restored Phase 138 summary authority after derived runs overwrote it:
  121 problems, 1907 workloads, 1761 passed, 146 failed, 86 OK, 35 FAIL.
- Isolated derived runs used `MemoryMax=20G` and `MemorySwapMax=0`; OOM-heavy
  problems were contained to their transient systemd user units and did not
  destabilize Codex.

## Boundaries

- Phase 140 does not upgrade timing, score, paper parity, leaderboard, CDNA3, or
  CDNA4 authority.
- Claim-upgrade and trust-summary reports preserve Phase 139 timing blockers.
- Consistency report records blocker findings for denominator/closure drift;
  Phase 141 must avoid stronger public claims until those findings are either
  resolved or explicitly accepted as a bounded artifact limitation.
- Derived sidecar exclusions are temporary local evidence and must remain
  visible/reversible.
