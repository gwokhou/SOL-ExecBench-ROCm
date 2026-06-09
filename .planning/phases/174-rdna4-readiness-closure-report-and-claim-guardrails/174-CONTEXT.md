---
phase: 174-rdna4-readiness-closure-report-and-claim-guardrails
created: 2026-06-09
source: gsd-autonomous interactive discussion
requirements:
  - COV-03
  - CLAIM-01
  - CLAIM-02
  - CLAIM-03
depends_on:
  - Phase 173
---

# Phase 174 Context: RDNA4 Readiness Closure Report and Claim Guardrails

## Goal

Finalize v1.34 by recomputing RDNA4 coverage across all 114 original
`readiness_blocked` problems, publishing blocker transition evidence, and
preserving claim-safe public/internal wording.

## Approved Decisions

1. Fixed audit set: use the 114 original `readiness_blocked` problems from
   `out/rdna4-coverage-current/coverage.json` as the fixed audit set. The final
   report must provide a disposition for every problem in that set.

2. Readiness is not validation: readiness reduction only means a problem became
   ready to attempt or received a more precise blocker class. It is not
   validation success, profiler-backed timing, paper parity, or leaderboard
   readiness unless execution evidence independently supports that claim.

3. Final ledger classes: the Phase 174 final blocker ledger must distinguish
   `resolved_readiness`, `execution_pass`, `execution_fail`, `oom`, `runtime`,
   `correctness`, `profiler`, `hardware_evidence_needed`, and
   `residual_readiness`.

4. RDNA4 evidence path: the RDNA4 environment is confirmed available. Phase 174
   should execute the real RDNA4 recompute, smoke, or coverage evidence path.
   If a specific run fails, classify the concrete failure as OOM, runtime,
   correctness, profiler, resource limit, or another precise class. Do not
   pre-classify the closure as `execution_environment_unavailable`.

5. Claim boundaries: documentation guardrails must state that this milestone
   does not upgrade CDNA3 or CDNA4 claims, does not change the 235-problem
   denominator, and does not mutate original or migrated dataset definitions or
   workloads.

## Implementation Implications

- The final closure report should be based on derived evidence and generated
  ledgers, not edits to dataset source artifacts.
- Any remaining original readiness-blocked problem must have a final residual
  class and next action.
- Reports should separate movement out of readiness blocking from actual
  execution outcomes.
- CPU-safe regression tests should prevent blocker loss, double counting, and
  accidental promotion to passed validation or profiler-backed timing.

