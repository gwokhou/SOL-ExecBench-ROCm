# Phase 171: Custom Input Coverage Recompute - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution
> agents. Decisions are captured in CONTEXT.md -- this log preserves the
> alternatives considered.

**Date:** 2026-06-09
**Phase:** 171-Custom Input Coverage Recompute
**Areas discussed:** Baseline Source, Transition Granularity, Attempt Boundary,
Residual Classification, Success Metric

---

## Baseline Source

| Option | Description | Selected |
| --- | --- | --- |
| Fixed current baseline | Use `out/rdna4-coverage-current/coverage.json` as the v1.34 before baseline and record path/checksum. | yes |
| Latest artifact auto-discovery | Let the latest coverage artifact replace the baseline. | no |

**User's choice:** Approved recommended decision.
**Notes:** Fixed baseline keeps before/after accounting reproducible.

---

## Transition Granularity

| Option | Description | Selected |
| --- | --- | --- |
| Problem plus optional workload | Require problem-level ledger; include workload-level transitions when evidence exists. | yes |
| Problem-only | Record only problem-level transitions. | no |

**User's choice:** Approved recommended decision.
**Notes:** Missing workload-level evidence must be explicit as
`workload_transition_unavailable`.

---

## Attempt Boundary

| Option | Description | Selected |
| --- | --- | --- |
| Bounded attempt with fallback | Try a small RDNA4 smoke if available; otherwise use CPU-safe simulated execution closure. | yes |
| Report-only | Recompute readiness/coverage without any attempt path. | no |
| RDNA4-required | Require real RDNA4 execution for phase success. | no |

**User's choice:** Approved recommended decision.
**Notes:** The hard gate is recompute plus complete transition ledger, not GPU
availability.

---

## Residual Classification

| Option | Description | Selected |
| --- | --- | --- |
| Explicit residual classes | Every unresolved original custom-input problem gets a specific residual class. | yes |
| Generic readiness blocked | Allow unresolved problems to remain generic `readiness_blocked`. | no |

**User's choice:** Approved recommended decision.
**Notes:** Required classes include unsupported entrypoint, gen-input OOM,
schema mismatch, device mismatch, timeout, and execution environment
unavailable.

---

## Success Metric

| Option | Description | Selected |
| --- | --- | --- |
| Complete disposition | Phase passes when all 55 original custom-input blockers have explicit dispositions. | yes |
| Count reduction required | Phase passes only if readiness-blocked count decreases. | no |

**User's choice:** Approved recommended decision.
**Notes:** Correct transitions may become OOM/runtime/correctness/profiler or
environment blockers, so count reduction is not a hard gate.

## the agent's Discretion

- Exact ledger filenames and schema.
- Additional deterministic residual classes if tested and documented.

## Deferred Ideas

- Quant and FlashInfer transition ledgers are deferred to later phases.

