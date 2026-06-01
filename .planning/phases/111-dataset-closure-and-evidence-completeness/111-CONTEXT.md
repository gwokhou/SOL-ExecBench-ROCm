# Phase 111 Context: Dataset Closure And Evidence Completeness

## Objective

Centralize dataset execution-closure record assembly for selected workloads so
trace references, summary references, solution references, derived evidence
references, evidence gaps, and missing-trace states are produced by package
helpers instead of duplicated script-local logic.

## Inputs

- Phase 110 extracted dataset reuse decisions into
  `sol_execbench.core.dataset.run_closure`.
- `scripts/run_dataset.py` still constructs attempted/skipped workload closure
  records inline in both reuse and fresh execution paths.
- Existing contracts use `sol_execbench.execution_closure.v1`; this phase must
  preserve that schema.

## Scope

- Add a core helper that builds a selected-workload closure record from a trace,
  evidence paths, and output refs.
- Use the helper in both reuse and fresh execution paths.
- Add CPU-safe tests for missing trace and missing derived evidence behavior.

## Out Of Scope

- New execution-closure schema fields.
- Live ROCm/GPU validation.
- Dataset sharding or merge behavior; that remains Phase 113.
