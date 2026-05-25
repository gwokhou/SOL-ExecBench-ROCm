---
gsd_state_version: 1.0
milestone: v1.14
milestone_name: Optional rocprofv3 Profiling Evidence
status: complete
last_updated: "2026-05-25T07:57:35Z"
last_activity: 2026-05-25
progress:
  total_phases: 3
  completed_phases: 3
  total_plans: 3
  completed_plans: 3
  percent: 100
---

# Project State

## Project Reference

See: `.planning/PROJECT.md` (updated 2026-05-25)

**Core value:** Evaluate LLM-generated GPU kernels correctly and reproducibly on AMD ROCm hardware while preserving the benchmark semantics and rigor of SOL ExecBench.
**Current focus:** v1.14 optional rocprofv3 profiling evidence

## Current Position

Phase: 63 complete
Plan: 63-PLAN.md
Status: Milestone complete
Last activity: 2026-05-25 — v1.14 profiling evidence milestone shipped

## Performance Metrics

**Velocity:**

- Total plans completed: 31
- Average duration: n/a
- Total execution time: 0.0 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 47. Derivation Contract And Golden Fixture Matrix | 6 | - | - |
| 48. Extraction Pipeline And Semantic Provenance | 4 | - | - |
| 49. High-Confidence Family Modeling | 4 | - | - |
| 50. Degraded Complex Family Modeling | 3 | - | - |
| 51. Sidecar Coverage And Score Guards | 3 | - | - |
| 52. Dataset Runner And Public Contract Closure | 3 | - | - |
| 53. Dataset Contract And Acquisition Metadata | 3 | - | - |
| 54. Paper Inventory And ROCm Readiness Classification | 3 | - | - |
| 55. Ready Subset Selection And Bounded Execution Closure | 3 | - | - |
| 56. Parity Gap Reporting And Evidence Review | 3 | - | - |
| 57. Claim Guardrails, Docs, And Release Closure | 2 | - | - |

**Recent Trend:**

- Last 5 plans: n/a
- Trend: n/a

*Updated after each plan completion*
| Phase 49 P01 | 6min | 3 tasks | 3 files |
| Phase 49 P02 | 4min | 2 tasks | 4 files |
| Phase 49 P03 | 7min | 2 tasks | 6 files |
| Phase 49 P04 | 5min | 3 tasks | 8 files |
| Phase 50 P01 | 8min | 3 tasks | 6 files |
| Phase 50 P02 | 11min | 3 tasks | 6 files |
| Phase 50 P03 | 6min | 3 tasks | 4 files |
| Phase 51 P01 | 5min | 3 tasks | 3 files |
| Phase 51 P02 | 6min | 3 tasks | 4 files |
| Phase 51 P03 | 5min | 3 tasks | 4 files |
| Phase 52 P01 | 6min | 3 tasks | 5 files |
| Phase 52 P03 | 31min | 3 tasks | 3 files |

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md. Recent decisions affecting current work:

- v1.10 scope is SOLAR derivation only.
- Preserve canonical trace JSONL, public schemas, primary CLI behavior, and AMD-native-derived claim boundaries.
- Defer 124-model/235-problem extraction, MI300X/CDNA3/CDNA4 validation, NVFP4/MXFP4 validation, hosted leaderboard, NVIDIA Blackwell/B200 equivalence, and new framework dependencies.
- 48-01 kept SOLAR derivation evidence internal and sidecar-only with explicit source_boundary booleans.
- 48-01 reused the existing EstimateConfidence vocabulary while serializing confidence as JSON-safe strings.
- 48-02 builds SOLAR derivation evidence only from Definition, Workload, BoundGraph, and OperatorWorkEstimate inputs.
- 48-02 keeps candidate solution execution explicitly outside the evidence builder boundary.
- 48-03 keeps SOLAR confidence classification pure and maps supported, inexact, and unsupported evidence to scored, degraded, and unscored states conservatively.
- 48-03 uses graph semantics for subrole provenance while retaining estimate provenance at the semantic-group level.
- [Phase 48]: Keep Phase 47 fixture expectations as data-only inputs and round-trip them through Phase 48 evidence without executing fixture references.
- [Phase 48]: Add required_evidence to the internal SOLAR semantic group sidecar for exact fixture-contract preservation while keeping public schemas unchanged.
- [Phase 48]: Keep Phase 48 evidence names and CLI switches excluded from canonical public schemas, trace JSONL, and primary sol-execbench help.
- [Phase 49]: Supported GEMM-compatible estimates record axis_source=tensor_shapes so SOLAR confidence can distinguish complete shape provenance from incomplete metadata.
- [Phase 49]: Visible linear projection bias tensors are represented as group-local subrole evidence without changing canonical schemas or score eligibility.
- [Phase 49]: Linear projection remains op_family=linear_projection while using formula_kind=gemm_flops and formula=2*M*N*K.
- [Phase 49]: Attention recognition stays inside the existing bound graph, estimate, and internal sidecar stack with no public schema changes.
- [Phase 49]: Partial mask tensors degrade attention evidence with mask:semantics and mask:sparsity rather than fabricating mask semantics.
- [Phase 49]: Direct q/k/v tensor inputs are represented as attention projection subroles when the surrounding QK, softmax, and PV structure is statically visible.
- [Phase 49]: Convolution and embedding/positional family evidence stays internal to SOLAR sidecars with selected-byte lookup estimates.
- [Phase 50]: MoE estimates use deterministic formula kinds `moe_static_route_flops` and `moe_dynamic_route_bytes`.
- [Phase 50]: Top-k, expert count, token count, and hidden size are included only when visible from parsed constants or tensor shapes.
- [Phase 50]: Taxonomy-only MoE calls remain unscored with `unsupported_operator:moe_taxonomy_only`.
- [Phase 50]: SSM/Mamba state update evidence requires visible state shape and update parameters.
- [Phase 50]: Opaque custom scan calls remain unscored with scan evidence but no fabricated recurrence metadata.
- [Phase 50]: SSM/Mamba scan estimates use deterministic formula kinds `ssm_mamba_static_scan_flops` and `ssm_mamba_degraded_scan_bytes`.
- [Phase 50]: Phase 50 internal formula kinds and warning names remain sidecar-only and absent from public schemas, primary CLI help, and canonical trace JSONL.
- [Phase 50]: Degraded MoE and SSM/Mamba evidence does not add SOLAR sidecar references to AMD-native score eligibility.
- [Phase 51]: SOLAR coverage and aggregate status are computed from existing semantic groups and warnings only.
- [Phase 51]: Phase 48-50 sidecar payloads remain parseable by exact legacy top-level keys, with Phase 51 fields recomputed on serialization.
- [Phase 51]: Degraded SOLAR aggregate status keeps numeric AMD-native scoring when numeric inputs are complete — Plan 51-02 REPORT-03 score guard
- [Phase 51]: Explicit SOLAR aggregate status unscored suppresses AMD-native score output — Plan 51-02 REPORT-03 score guard
- [Phase 51]: Keep Phase 51 coverage and aggregate fields sidecar-only; do not add public schemas, CLI flags, report fields, dependencies, GPU checks, or candidate execution.
- [Phase 51]: Treat AMD SOL v2 coverage_summary as an existing artifact field while guarding exact SOLAR sidecar keys from leaking into v2 payloads.
- [Phase 52]: Derived report audit refs live in AmdNativeScore.derived_evidence_refs, leaving public evidence_refs keys unchanged.
- [Phase 52]: The dataset-runner skip branch builds requested derived artifacts before continuing when existing traces already pass.
- [Phase 52]: Generated SOLAR derivation sidecars are built and immediately parsed through the existing sidecar parser before scoring.

### Pending Todos

None found.

### Blockers/Concerns

None active.

### Quick Tasks Completed

| # | Description | Date | Commit | Directory |
|---|-------------|------|--------|-----------|
| 260524-xb3 | Fix PR base and prepare SOL evaluator contract branch for merge | 2026-05-24 | 5d4892d | [260524-xb3-fix-pr-base-and-prepare-sol-evaluator-co](./quick/260524-xb3-fix-pr-base-and-prepare-sol-evaluator-co/) |
| 260525-097 | Backfill v1.12 GSD milestone for evaluator contract PR | 2026-05-25 | this commit | [260525-097-backfill-v1-12-gsd-milestone-for-evaluat](./quick/260525-097-backfill-v1-12-gsd-milestone-for-evaluat/) |

## Deferred Items

Items acknowledged and carried forward from previous milestone close:

| Category | Item | Status | Deferred At |
|----------|------|--------|-------------|
| Dataset extraction | Original paper 124-model / 235-problem extraction | Deferred | v1.10 scope |
| Hardware validation | MI300X, CDNA 3, and CDNA 4 real-hardware validation | Deferred | v1.10 scope |
| Hardware validation | NVFP4 and MXFP4 validation | Deferred | v1.10 scope |
| Public service | Hosted leaderboard or submission service | Deferred | v1.10 scope |
| Claims | NVIDIA Blackwell/B200 equivalence | Deferred | v1.10 scope |

## Session Continuity

Last session: 2026-05-23T11:26:49.344Z
Stopped at: Completed 52-03-PLAN.md
Resume file: None

## Operator Next Steps

- Start the next milestone with /gsd-new-milestone
