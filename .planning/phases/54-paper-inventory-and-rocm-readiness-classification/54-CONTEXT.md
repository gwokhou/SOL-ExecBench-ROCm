# Phase 54: Paper Inventory And ROCm Readiness Classification - Context

**Gathered:** 2026-05-23
**Status:** Ready for planning

<domain>
## Phase Boundary

This phase generates deterministic paper parity inventory and ROCm readiness
classification artifacts for the public dataset surface. It reads canonical
`definition.json` and `workload.jsonl` files through the current Pydantic
contracts, records schema/layout/asset metadata, classifies static readiness
for local ROCm execution attempts, and emits a ready-subset sidecar. It does
not run GPU evaluation, mutate canonical dataset files, claim execution pass,
claim scoring success, or claim paper parity.

</domain>

<decisions>
## Implementation Decisions

### Inventory Scope And Parsing Boundary
- Use the Phase 53 dataset root and manifest as entry context, but re-read
  canonical `definition.json` and `workload.jsonl` files as the source of truth.
- Schema parsing failures should produce structured `schema_failure` records
  and denominator counts without aborting the whole inventory run.
- Record reference and solution availability by path only; do not import,
  execute, or inspect runtime behavior.
- Model inventory as problem-level records with workload-level child records,
  including UUIDs, input kinds, dtype/shape hints, custom input usage, and
  safetensors usage at workload granularity.

### ROCm Readiness Status Model
- Classify readiness primarily at workload level, with problem-level summaries
  derived from the most blocking workload states.
- Define `ready` as "ready to attempt local ROCm execution", not as validated
  or passed. It requires parseable schema, locatable/generatable inputs,
  reference availability, and no known NVIDIA-only/runtime blocker.
- Represent low-precision and Quant readiness in layers: schema-known,
  input-generation, reference-execution, candidate-execution, and
  hardware-validation evidence.
- Treat custom inputs and safetensors as explicit asset/evaluator requirements;
  missing local assets or unsupported evaluator paths become blockers rather
  than random substitution.

### Artifact Shape And CLI/API Surface
- Extend `src/sol_execbench/core/dataset/` with inventory and readiness modules;
  keep scripts as thin CLI wrappers.
- Emit deterministic `inventory.json`, `readiness.json`, and
  `ready_subset.json` sidecar artifacts with schema versions.
- Add a thin `scripts/inspect_dataset.py` CLI supporting `--dataset-root`,
  `--manifest`, `--inventory`, `--readiness`, and `--ready-subset`.
- Re-export stable helper/model names from `sol_execbench.core.dataset` only;
  do not change the primary `sol-execbench` CLI or public benchmark schemas.

### Verification, Fixtures, And Guardrails
- Use fixture tests covering schema-ok ready, schema failure, missing files,
  custom inputs, safetensors, unsupported dtype/Quant, NVIDIA-only hints, and
  hardware-evidence-needed cases.
- Do not run real GPU/ROCm evaluation in Phase 54.
- Generate ready-subset manifests from readiness results only, including stable
  references for `ready` workloads and no canonical dataset mutations.
- Add tests and docs wording that readiness means "can attempt execution", not
  passed, scored, fully validated, or paper-parity complete.

### the agent's Discretion
The agent may choose exact module names, enum/model shapes, status severity
ordering, and CLI formatting as long as outputs are deterministic, sidecar-only,
and preserve the locked scope boundaries.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- Phase 53 added `src/sol_execbench/core/dataset/` for category validation,
  layout inspection, deterministic manifests, and claim boundaries.
- `src/sol_execbench/core/data/definition.py` and
  `src/sol_execbench/core/data/workload.py` are the authoritative Pydantic
  contracts for inventory parsing.
- `scripts/download_solexecbench.py` can emit dataset manifests for the local
  root; Phase 54 can consume those manifests as provenance.

### Established Patterns
- Keep derived metadata in sidecar JSON artifacts.
- Use deterministic ordering and stable schema-version fields for machine
  artifacts.
- Use fixture/mocked tests rather than real network or GPU work for dataset
  management behavior.

### Integration Points
- New inventory/readiness helpers belong under `src/sol_execbench/core/dataset/`.
- A new thin script should live in `scripts/inspect_dataset.py`.
- Public guardrails should extend the Phase 53 dataset contract tests and
  existing public contract guardrail tests.

</code_context>

<specifics>
## Specific Ideas

Prioritize paper-aligned denominators and explicit blockers over optimistic
readiness. Unknown or unsupported evidence should be machine-readable and
actionable rather than collapsed into a generic failure.

</specifics>

<deferred>
## Deferred Ideas

- Ready-subset execution is deferred to Phase 55.
- Parity gap report aggregation is deferred to Phase 56.
- Milestone release claim closure is deferred to Phase 57.
- Full public dataset real-hardware validation remains out of scope.

</deferred>
