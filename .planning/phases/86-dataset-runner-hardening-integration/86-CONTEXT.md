# Phase 86: Dataset Runner Hardening Integration - Context

**Gathered:** 2026-05-31
**Status:** Ready for planning

<domain>
## Phase Boundary

This phase hardens `scripts/run_dataset.py` resume/reuse behavior by reusing
Phase 83 execution-closure helpers to classify provenance-safe reuse, rerun
attempts, skipped workloads, failures, missing traces, and missing derived
evidence. It must preserve existing default execution behavior unless an
existing-output reuse path, mismatch, missing evidence, or explicit closure
option requires diagnostic closure status.

</domain>

<decisions>
## Implementation Decisions

### Default Behavior And Enforcement
- Preserve runner defaults as much as possible; do not change normal fresh
  execution behavior.
- Enable stricter provenance checks when resuming or reusing existing output.
- Existing traces may be treated as reusable only when ready-subset, readiness,
  manifest, problem/workload identity, solution mode, and requested evidence
  requirements match the selected run configuration.
- On provenance mismatch, record diagnostic closure status and avoid treating
  the old trace as a reusable pass.

### Classification And Bounded Logs
- Reuse Phase 83 closure helpers and stable vocabularies where practical.
- Classify build failures, runtime failures, timeouts, nonzero CLI exits,
  correctness failures, missing traces, missing derived evidence, skipped, and
  unattempted workloads deterministically.
- Keep log refs bounded and relative; do not embed credentials, proprietary
  kernels, raw dataset payloads, unnecessary absolute paths, or unbounded logs.
- Preserve deterministic record ordering and checksummed sidecar output.

### Rerun And Existing Pass Reuse
- `skipped_existing_pass` is allowed only when provenance matches and `--rerun`
  is not set.
- `--rerun` must force a fresh attempt and record the fresh attempt result or
  failure classification in execution closure.
- A passing trace alone is insufficient for `skipped_existing_pass` if
  provenance is missing or mismatched.
- Missing requested derived evidence should remain `derived_evidence_missing`
  or another explicit diagnostic state, not a reusable clean pass.

### the agent's Discretion
- The agent may choose exact helper boundaries and whether to extend
  `execution_closure.py` or keep runner-specific wrappers in
  `scripts/run_dataset.py`, but the default behavior compatibility constraint
  is stronger than abstraction cleanup.
- The agent may add narrow reason codes if required for runner failure
  classification, provided they remain stable, bounded, and covered by tests.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/sol_execbench/core/dataset/execution_closure.py` provides strict
  execution closure models, status helpers, provenance comparison, totals,
  deterministic serialization, and checksum handling.
- `scripts/run_dataset.py` already delegates closure report construction and
  provenance fields to Phase 83 helpers.
- `tests/sol_execbench/test_run_dataset_execution_closure.py` contains CPU-safe
  runner tests for ready subsets, readiness blockers, derived evidence gaps,
  sorted closure records, existing pass skipping, and bounded refs.
- `tests/sol_execbench/test_execution_closure_contract.py` covers closure helper
  strictness and provenance comparisons.

### Established Patterns
- Runner tests monkeypatch `run_cli` and related helpers rather than invoking
  ROCm, Docker, or real GPU execution.
- Existing failure logs are written as files under output directories; closure
  sidecars should reference bounded relative log paths rather than embedding
  full stdout/stderr.
- The current existing-pass path is around `scripts/run_dataset.py` lines
  1418-1480 and needs careful treatment to avoid default regressions.

### Integration Points
- Focus changes around existing-output reuse, `--rerun`, `run_cli` failure
  classification, and closure record construction.
- Keep public canonical Trace, Definition, Workload, Solution, timing, score,
  and evaluator semantics unchanged.
- Future Phase 88 will document the final behavior, so reason codes and closure
  notes should be stable and concise.

</code_context>

<specifics>
## Specific Ideas

Plan should start with targeted tests for existing-pass provenance mismatch,
`--rerun` fresh attempts, missing traces, and bounded CLI log refs. Then make
the smallest runner changes needed to pass those tests.

</specifics>

<deferred>
## Deferred Ideas

- New hardware validation, native-host validation, CDNA 3, MI300X, and CDNA 4
  validation remain out of scope.
- Paper denominator and Matrix reporting changes are complete in prior phases
  and should not be reworked here.
- Hosted leaderboard or CI policy gates remain future work.

</deferred>
