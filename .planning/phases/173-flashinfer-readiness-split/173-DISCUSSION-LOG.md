# Phase 173: FlashInfer Readiness Split - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution
> agents. Decisions are captured in CONTEXT.md -- this log preserves the
> alternatives considered.

**Date:** 2026-06-09
**Phase:** 173-FlashInfer Readiness Split
**Areas discussed:** Simple Case Policy, Runtime Buckets, Evidence Required,
Adaptation Boundary, Execution Boundary

---

## Simple Case Policy

| Option | Description | Selected |
| --- | --- | --- |
| Static simple release | Release PyTorch-only migrated `rmsnorm`, `fused_add_rmsnorm`, and plain GEMM when no runtime metadata/imports exist. | yes |
| Keep category-wide block | Continue blocking all FlashInfer-Bench problems by category. | no |

**User's choice:** Approved recommended decision.
**Notes:** Static release requires evidence beyond problem name.

---

## Runtime Buckets

| Option | Description | Selected |
| --- | --- | --- |
| Fixed six buckets | Use `paged_decode`, `paged_prefill`, `ragged_prefill`, `mla_paged`, `moe_fp8_block_scale`, and `unknown_flashinfer_runtime`. | yes |
| Open-ended labels only | Let each problem define arbitrary labels. | no |

**User's choice:** Approved recommended decision.
**Notes:** Phase 173 must cover all 26 current FlashInfer-Bench problems.

---

## Evidence Required

| Option | Description | Selected |
| --- | --- | --- |
| Structured release evidence | Record problem id, semantic bucket, migrated reference path, no-import/call evidence, schema compatibility, and rationale. | yes |
| Problem-name heuristic | Classify from name only. | no |

**User's choice:** Approved recommended decision.
**Notes:** Problem name alone is not sufficient.

---

## Adaptation Boundary

| Option | Description | Selected |
| --- | --- | --- |
| Reuse local migration path | Update migration/classifier/derived evidence and regenerate locally if metadata is missing. | yes |
| Hand patch migrated files | Edit generated benchmark artifacts in place. | no |
| Raw HF runner input | Make runner consume original Hugging Face dataset directly. | no |

**User's choice:** Approved updated recommendation.
**Notes:** Current FlashInfer-Bench layout is migrated/generated from
`flashinfer-ai/flashinfer-trace`, not an untouched raw HF dataset structure.

---

## Execution Boundary

| Option | Description | Selected |
| --- | --- | --- |
| Classification required, smoke optional | Complete semantic ledger; smoke newly-ready cases if environment allows. | yes |
| Execution required | Require real smoke execution for phase success. | no |

**User's choice:** Approved recommended decision.
**Notes:** Execution unavailability should be explicit evidence, not validation.

## the agent's Discretion

- Exact taxonomy helper names.
- Evidence schema and location.

## Deferred Ideas

- FlashInfer ROCm performance kernel tuning.
- Direct raw HF dataset execution.

