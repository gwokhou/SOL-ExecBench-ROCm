# Phase 174 Discussion Log

## 2026-06-09

User approved the default Phase 174 decision set with one correction:

- The RDNA4 environment is confirmed available.
- Phase 174 should therefore use the real RDNA4 evidence path rather than
  defaulting to `execution_environment_unavailable`.

Final approved decisions:

1. Fix the audit set to the 114 original `readiness_blocked` problems from
   `out/rdna4-coverage-current/coverage.json`.
2. Treat readiness reduction as readiness movement only, not validation success.
3. Require final ledger categories for readiness, execution, OOM, runtime,
   correctness, profiler, hardware-evidence, and residual blockers.
4. Execute real RDNA4 recompute, smoke, or coverage evidence where required;
   classify concrete failures precisely.
5. Preserve claim boundaries for CDNA3, CDNA4, denominator policy, and dataset
   immutability.

