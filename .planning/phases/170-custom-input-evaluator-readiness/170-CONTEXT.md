# Phase 170: Custom Input Evaluator Readiness - Context

**Gathered:** 2026-06-09
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 170 implements deterministic benchmark-defined custom input generation so
the 55 L1/L2 custom-input readiness blockers can be attempted safely on ROCm.
This phase is evaluator/input-assembly work only: it must not mutate original
or migrated dataset definitions, references, or workload rows. Generated inputs,
classification changes, and provenance are derived execution evidence.

</domain>

<decisions>
## Implementation Decisions

### Entrypoint Scope

- **D-01:** Support `custom_inputs_entrypoint` declared in `definition.json`.
  Load the implementation from `reference.py` first when present.
- **D-02:** Also support definitions where the reference implementation is
  available only as an inline reference string in `definition.json`.
- **D-03:** Do not proactively add alias or legacy custom-entrypoint formats
  unless existing inventory parsing can identify them unambiguously.

### Determinism Policy

- **D-04:** Use a stable per-workload seed derived from problem id plus workload
  UUID or row index.
- **D-05:** Isolate and restore PyTorch RNG state around custom input
  generation so benchmark input generation does not leak randomness into later
  execution.
- **D-06:** Record enough seed/provenance evidence for reproducible RDNA4
  closure reruns.

### Validation Strictness

- **D-07:** Validate generated inputs before reference or candidate execution.
- **D-08:** Treat missing required keys, invalid extra required keys, dtype
  mismatches, shape mismatches, device mismatches, and tensor/scalar kind
  mismatches as input-generation blockers.
- **D-09:** Resolve dynamic shapes through workload axes before validation.
  Reference/candidate execution should not be used as the first line of schema
  validation for generated inputs.

### Failure Classes

- **D-10:** Preserve fine-grained input-generation failure classes in evidence:
  `gen_inputs_error`, `gen_inputs_oom_blocked`, `gen_inputs_timeout`,
  `gen_inputs_schema_mismatch`, and `gen_inputs_device_mismatch`.
- **D-11:** Reports may aggregate these classes for summary readability, but
  the lower-level evidence must retain the specific class.

### Execution Boundary

- **D-12:** Phase 170 must complete with CPU-safe unit and synthetic fixture
  coverage. Real RDNA4 smoke execution is optional evidence if the local GPU
  environment is available.
- **D-13:** Phase 170 passing criteria must not depend on real RDNA4 execution;
  broader RDNA4 coverage movement belongs to Phase 171.

### Dataset Boundary

- **D-14:** Custom input support is runner/evaluator behavior only. It must not
  rewrite upstream original dataset files or locally migrated dataset
  definitions/workloads.
- **D-15:** If a dataset entry is malformed, record an explicit blocker or
  derived sidecar/evidence record instead of silently modifying the dataset.

### the agent's Discretion

The agent may choose the exact helper/module boundaries, fixture layout, and
sidecar schema fields as long as the locked decisions above are preserved and
nearby dataset/eval-driver patterns are followed.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Milestone and Phase Scope

- `.planning/PROJECT.md` - Current v1.34 milestone goal, target features, and
  claim boundaries.
- `.planning/REQUIREMENTS.md` - Requirements `CUST-01` through `CUST-04`.
- `.planning/ROADMAP.md` - Phase 170 goal, deliverables, dependencies, and
  success criteria.
- `.planning/research/SUMMARY.md` - Recommended build order and readiness
  blocker research summary.

### Codebase Maps

- `.planning/codebase/ARCHITECTURE.md` - Evaluation, dataset, staging, and
  runtime layers.
- `.planning/codebase/INTEGRATIONS.md` - PyTorch ROCm, dataset assets, user code
  integration, and process boundaries.
- `.planning/codebase/TESTING.md` - Testing patterns, marker gates, artifact
  assertions, and CPU-safe fixture strategy.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets

- `src/sol_execbench/core/data/definition.py`: schema source for definition
  metadata, including custom input entrypoint fields.
- `src/sol_execbench/core/data/workload.py`: workload input descriptors,
  custom inputs, axes, tolerance, and concrete workload rows.
- `src/sol_execbench/core/bench/io.py`: input generation/loading helpers used
  inside the staged evaluator.
- `src/sol_execbench/core/bench/eval_runtime.py`: runtime helpers for staged
  reference/candidate evaluation.
- `src/sol_execbench/driver/templates/eval_driver.py`: staged subprocess
  boundary where custom generated inputs must be available before reference and
  candidate calls.
- `src/sol_execbench/core/dataset/readiness.py`: static readiness blocker
  classification currently marking custom-input workloads blocked.
- `scripts/run_dataset.py`: operational dataset runner that will consume
  readiness and execution closure changes.

### Established Patterns

- Runtime behavior that must execute inside staged evaluation belongs under
  `src/sol_execbench/core/bench/` or the staged `eval_driver.py` template.
- Dataset classification/reporting behavior belongs under
  `src/sol_execbench/core/dataset/`.
- Tests should use real JSON/JSONL/source snippets and `tmp_path` fixtures,
  with CPU-safe coverage as the required baseline.
- Canonical trace JSONL should remain strict; derived readiness/provenance
  details should be sidecars or structured diagnostics.

### Integration Points

- Readiness should stop treating supported custom inputs as generic
  `readiness_blocked` once evaluator support exists.
- Eval input assembly must run custom input entrypoints before reference and
  candidate execution.
- Execution closure and coverage reports should consume the new input
  generation failure classes without changing the 235-problem denominator.

</code_context>

<specifics>
## Specific Ideas

- Benchmark semantics belong to the dataset-defined custom input entrypoint.
  GPU Kernel Agent integrations should submit kernels/solutions; the evaluator
  should provide stable, semantically correct inputs.
- Stable input generation makes agent benchmark runs reproducible and prevents
  input randomness from being mistaken for kernel quality differences.
- Strict pre-execution validation keeps input-generation, reference, and
  candidate failures separated for agent-facing diagnostics.

</specifics>

<deferred>
## Deferred Ideas

- Real RDNA4 coverage movement for the 55 custom-input blockers belongs to
  Phase 171 after Phase 170 evaluator support exists.
- Quant readiness and FlashInfer readiness are explicitly deferred to Phases
  172 and 173.
- High-performance kernel tuning and broader hardware validation remain out of
  scope for this phase.

</deferred>

---

*Phase: 170-Custom Input Evaluator Readiness*
*Context gathered: 2026-06-09*

