# Phase 172: Quant Readiness Triage - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution
> agents. Decisions are captured in CONTEXT.md -- this log preserves the
> alternatives considered.

**Date:** 2026-06-09
**Phase:** 172-Quant Readiness Triage
**Areas discussed:** Hint Detection Rule, Quant Outcome Classes, False Positive
Handling, Low Precision Boundary, Residual Evidence

---

## Hint Detection Rule

| Option | Description | Selected |
| --- | --- | --- |
| Context-aware true blockers | Only import/call/native source/solution dependency evidence blocks. | yes |
| Broad lexical matching | Continue blocking comments, class names, and variable names. | no |

**User's choice:** Approved recommended decision.
**Notes:** Compatibility labels should not overblock otherwise executable
semantic references.

---

## Quant Outcome Classes

| Option | Description | Selected |
| --- | --- | --- |
| Precise readiness/hardware split | Route to ready or needs-hardware-evidence based on dtype/format and PyTorch ROCm compatibility. | yes |
| Automatically ready | Mark Quant ready once CUDA lexical hints are cleared. | no |

**User's choice:** Approved recommended decision.
**Notes:** Low-precision evidence remains separate from execution readiness.

---

## False Positive Handling

| Option | Description | Selected |
| --- | --- | --- |
| Classifier-only fix | Do not mutate dataset/reference names; record derived false-positive evidence. | yes |
| Dataset rename | Rewrite migrated references to remove `cublas/cuda` words. | no |

**User's choice:** Approved recommended decision.
**Notes:** This preserves original/migrated dataset provenance.

---

## Low Precision Boundary

| Option | Description | Selected |
| --- | --- | --- |
| FP8 and NVFP4 split | FP8 may be RDNA4 ready/smoke-attempt; NVFP4/MXFP4/CDNA4 remains deferred/evidence-needed. | yes |
| One low-precision bucket | Treat all low precision formats identically. | no |

**User's choice:** Approved recommended decision.
**Notes:** Readiness movement must not imply CDNA4 validation.

---

## Residual Evidence

| Option | Description | Selected |
| --- | --- | --- |
| Source evidence required | Record problem id, file path, matched token, match kind, and line/context. | yes |
| Problem-level reason only | Store only high-level blocker reason. | no |

**User's choice:** Approved recommended decision.
**Notes:** Residual true CUDA blockers need actionable source evidence.

## the agent's Discretion

- Exact helper and evidence schema names.
- Additional match kinds when deterministic and tested.

## Deferred Ideas

- Quant kernel performance optimization.
- CDNA4 validation.

