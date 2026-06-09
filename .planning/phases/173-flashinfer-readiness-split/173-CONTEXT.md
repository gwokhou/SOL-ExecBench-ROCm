# Phase 173: FlashInfer Readiness Split - Context

**Gathered:** 2026-06-09
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 173 splits the 26 FlashInfer-Bench readiness blockers by semantic
dependency so simple PyTorch-compatible migrated references can be attempted
while true FlashInfer runtime workloads retain precise residual blockers. This
phase works through the existing local migration/readiness path and derived
evidence; it must not hand-edit migrated benchmark artifacts.

</domain>

<decisions>
## Implementation Decisions

### Simple Case Policy

- **D-01:** Allow PyTorch-only migrated references such as `rmsnorm`,
  `fused_add_rmsnorm`, and plain GEMM to move from category-wide
  `flashinfer_runtime_assumption` to `ready_to_attempt_rocm_execution`.
- **D-02:** A simple case may be released only when static evidence shows no
  FlashInfer import/call, no paged/ragged/MLA/MoE runtime metadata, and input
  schema compatibility with the existing runner.

### Runtime Buckets

- **D-03:** Use fixed residual buckets for the current 26 problems:
  `paged_decode`, `paged_prefill`, `ragged_prefill`, `mla_paged`,
  `moe_fp8_block_scale`, and `unknown_flashinfer_runtime`.
- **D-04:** Future buckets may be added later, but Phase 173 must classify all
  26 current FlashInfer-Bench problems into simple-ready or one of the residual
  buckets.

### Evidence Required

- **D-05:** Simple-case release evidence must include problem id, matched
  semantic bucket, migrated reference path, no-FlashInfer import/call evidence,
  input/workload schema compatibility, and the classification rationale.
- **D-06:** Problem name alone is not sufficient evidence for releasing a
  FlashInfer-Bench problem from category-wide blocking.

### Adaptation Boundary

- **D-07:** ROCm adaptation for FlashInfer-Bench should reuse the existing local
  migration path from original `flashinfer-ai/flashinfer-trace` data into the
  migrated `data/SOL-ExecBench/benchmark/FlashInfer-Bench` layout.
- **D-08:** If migrated artifacts are missing metadata needed for readiness
  classification, update migration logic, classifier logic, or derived evidence
  generation and regenerate locally.
- **D-09:** Do not hand-edit migrated benchmark artifacts and do not make the
  runner consume the original Hugging Face dataset directly.

### Execution Boundary

- **D-10:** Phase 173 must complete semantic classification and a residual
  blocker ledger for all 26 FlashInfer-Bench problems.
- **D-11:** A smoke attempt for newly-ready simple cases is recommended
  additional evidence but is not a hard phase gate.
- **D-12:** If execution is unavailable, record
  `execution_environment_unavailable` rather than treating the classification as
  execution validation.

### the agent's Discretion

The agent may choose exact taxonomy helper names, derived evidence schema, and
classifier integration points as long as classification is deterministic and
does not patch migrated dataset files by hand.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Milestone and Phase Scope

- `.planning/PROJECT.md` - v1.34 FlashInfer target feature and claim
  boundaries.
- `.planning/REQUIREMENTS.md` - Requirements `FLASH-01` through `FLASH-04`.
- `.planning/ROADMAP.md` - Phase 173 deliverables and success criteria.
- `.planning/research/SUMMARY.md` - FlashInfer readiness split research.

### Prior Context

- `.planning/phases/172-quant-readiness-triage/172-CONTEXT.md` - Prior phase
  classifier/evidence boundary decisions and no-dataset-mutation pattern.

### Local Data and Baseline Evidence

- `data/SOL-ExecBench/benchmark/FlashInfer-Bench` - Migrated local benchmark
  layout containing `definition.json`, `reference.py`, and `workload.jsonl`
  per problem.
- `out/rdna4-coverage-current/coverage.json` - Current coverage baseline
  showing 26 FlashInfer-Bench readiness blockers.

### Codebase Maps

- `.planning/codebase/ARCHITECTURE.md` - Dataset migration, inventory,
  readiness, and runner flow.
- `.planning/codebase/INTEGRATIONS.md` - FlashInfer Trace dataset source,
  safetensors roots, and local migration/run boundaries.
- `.planning/codebase/TESTING.md` - CPU-safe readiness and artifact test
  patterns.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets

- `src/sol_execbench/core/dataset/migration.py`: local dataset migration logic
  for generated benchmark layout.
- `src/sol_execbench/core/dataset/inventory.py`: inventory extraction over
  migrated problem definitions, references, and workloads.
- `src/sol_execbench/core/dataset/readiness.py`: current category/path-based
  FlashInfer runtime blocker logic.
- `scripts/run_dataset.py`: dataset execution path that uses migrated benchmark
  layout, not raw Hugging Face records.
- `scripts/download_data.sh`: documents raw FlashInfer Trace download to
  `data/flashinfer-trace`.

### Established Patterns

- Raw external datasets are downloaded locally, then transformed through
  project-owned migration logic into benchmark artifacts.
- Readiness classification should operate on migrated artifacts and derived
  evidence, not by mutating dataset payloads in place.
- Residual blockers should include next actions and evidence, not just category
  labels.

### Integration Points

- FlashInfer readiness classifier should switch from category-wide blocking to
  semantic bucket classification.
- Migration/classifier evidence may need additional metadata for no-import,
  schema-compatible, and runtime-dependent decisions.
- Phase 174 will consume the Phase 173 blocker ledger for final all-114 closure.

</code_context>

<specifics>
## Specific Ideas

- The current local FlashInfer-Bench directory is a migrated/generated benchmark
  layout, not the raw Hugging Face dataset structure.
- Any ROCm adaptation needed for readiness should be implemented in migration,
  classifier, or derived evidence generation so it is reproducible.

</specifics>

<deferred>
## Deferred Ideas

- High-performance FlashInfer-equivalent ROCm kernels are out of scope.
- Direct raw Hugging Face dataset execution is out of scope.
- Final readiness closure and public claim wording are deferred to Phase 174.

</deferred>

---

*Phase: 173-FlashInfer Readiness Split*
*Context gathered: 2026-06-09*

