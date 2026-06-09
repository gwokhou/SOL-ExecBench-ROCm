# Phase 170: Custom Input Evaluator Readiness - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution
> agents. Decisions are captured in CONTEXT.md -- this log preserves the
> alternatives considered.

**Date:** 2026-06-09
**Phase:** 170-Custom Input Evaluator Readiness
**Areas discussed:** Entrypoint Scope, Determinism Policy, Validation
Strictness, Failure Classes, Execution Boundary

---

## Entrypoint Scope

| Option | Description | Selected |
| --- | --- | --- |
| Definition/reference scope | Support `custom_inputs_entrypoint` from `definition.json`, loading `reference.py` first and inline reference strings as fallback. | yes |
| Expanded compatibility | Also support aliases or legacy formats proactively. | no |

**User's choice:** Approved recommended decision.
**Notes:** The decision preserves benchmark-defined input semantics without
inventing undocumented formats.

---

## Determinism Policy

| Option | Description | Selected |
| --- | --- | --- |
| Per-workload stable seed | Derive seed from problem id plus workload UUID or row index, isolate/restore PyTorch RNG state, and record provenance. | yes |
| Runner-global seed only | Reuse a single runner seed without per-workload derivation. | no |
| Record only | Record randomness but do not enforce deterministic state. | no |

**User's choice:** Approved recommended decision.
**Notes:** Determinism is important for application-level GPU Kernel Agent
benchmark integration.

---

## Validation Strictness

| Option | Description | Selected |
| --- | --- | --- |
| Strict pre-execution validation | Validate keys, tensor/scalar kinds, dtypes, shapes, and devices before reference/candidate execution. | yes |
| Reference-driven validation | Let reference/candidate execution surface most mismatches. | no |

**User's choice:** Approved recommended decision.
**Notes:** Strict validation keeps input-generation failures separate from
kernel/reference failures.

---

## Failure Classes

| Option | Description | Selected |
| --- | --- | --- |
| Fine-grained classes | Preserve `gen_inputs_error`, `gen_inputs_oom_blocked`, `gen_inputs_timeout`, `gen_inputs_schema_mismatch`, and `gen_inputs_device_mismatch`. | yes |
| Minimal classes | Only split generic error and OOM. | no |

**User's choice:** Approved recommended decision.
**Notes:** Reports may aggregate classes, but lower-level evidence must retain
specific classification.

---

## Execution Boundary

| Option | Description | Selected |
| --- | --- | --- |
| CPU-safe required, RDNA4 optional | Phase passes on unit/synthetic fixture coverage; real RDNA4 smoke is optional if available. | yes |
| RDNA4 required | Phase cannot pass without real RDNA4 custom-input smoke. | no |

**User's choice:** Approved recommended decision.
**Notes:** Phase 170 implements evaluator capability. RDNA4 coverage movement
belongs to Phase 171.

---

## Dataset Boundary Clarification

**User question:** Whether the decisions are natural for application-level GPU
Kernel Agent benchmark integration and how they affect the original dataset.

**Captured decision:** Custom input support must be runner/evaluator behavior
only. It must not mutate original or migrated dataset definitions/workloads; all
generated inputs and classifications are derived execution evidence.

## the agent's Discretion

- Exact helper/module boundaries.
- Fixture layout.
- Sidecar field names, as long as evidence remains structured and claim-safe.

## Deferred Ideas

- Real RDNA4 custom-input coverage movement is deferred to Phase 171.
- Quant and FlashInfer readiness are deferred to Phases 172 and 173.

