# Phase 171: Custom Input Coverage Recompute - Context

**Gathered:** 2026-06-09
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 171 recomputes RDNA4 readiness and coverage after Phase 170 custom input
support. It must account for all 55 original custom-input readiness blockers
and show whether each moved to ready, pass/fail, runtime/OOM, profiler,
environment, or residual readiness states. This phase does not add new custom
input evaluator capability; it consumes Phase 170 behavior and reports the
coverage transition.

</domain>

<decisions>
## Implementation Decisions

### Baseline Source

- **D-01:** Use `out/rdna4-coverage-current/coverage.json` as the fixed v1.34
  before baseline for custom-input transition accounting.
- **D-02:** Record the baseline path and checksum in the transition ledger.
- **D-03:** The latest coverage artifact may be recorded as supplementary
  context, but it must not replace the fixed before baseline.

### Transition Granularity

- **D-04:** Produce a problem-level transition ledger for all 55 original
  custom-input readiness blockers.
- **D-05:** Also produce workload-level transitions when Phase 170 generated
  workload-level evidence is available.
- **D-06:** If workload-level evidence is unavailable for a problem, mark that
  explicitly as `workload_transition_unavailable` instead of implying workload
  closure.

### Attempt Boundary

- **D-07:** Include a bounded attempt path for newly-ready custom-input
  problems. Prefer a small RDNA4 smoke if the environment is available; otherwise
  use CPU-safe simulated execution closure.
- **D-08:** Real RDNA4 execution is not a hard phase dependency. The hard gate is
  readiness/coverage recompute plus a complete transition ledger.
- **D-09:** Any bounded attempt must preserve claim boundaries: readiness or
  smoke movement is not full validation success.

### Residual Classification

- **D-10:** Every original custom-input problem that is not safely executable
  must receive a residual class; generic `readiness_blocked` is insufficient.
- **D-11:** Required residual classes include at least
  `unsupported_custom_entrypoint`, `gen_inputs_oom_blocked`,
  `gen_inputs_schema_mismatch`, `gen_inputs_device_mismatch`,
  `gen_inputs_timeout`, and `execution_environment_unavailable`.
- **D-12:** Additional residual classes are allowed if they are deterministic,
  documented, and included in tests.

### Success Metric

- **D-13:** Phase 171 succeeds when all 55 original custom-input readiness
  blockers have an explicit disposition.
- **D-14:** A lower `readiness_blocked` count is a desired outcome but not a hard
  pass condition, because correct transitions may become OOM, runtime,
  correctness, profiler, hardware, or environment blockers.

### the agent's Discretion

The agent may choose exact filenames and schemas for transition ledgers as long
as the baseline is fixed, problem-level accounting is complete, and
workload-level availability is explicit.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Milestone and Phase Scope

- `.planning/PROJECT.md` - v1.34 milestone scope and claim boundaries.
- `.planning/REQUIREMENTS.md` - Requirements `COV-01` and `COV-02`.
- `.planning/ROADMAP.md` - Phase 171 deliverables and success criteria.
- `.planning/research/SUMMARY.md` - Recommended sequencing and reporting
  pitfalls.

### Prior Phase Context

- `.planning/phases/170-custom-input-evaluator-readiness/170-CONTEXT.md` -
  Custom input evaluator decisions that Phase 171 consumes.

### Baseline Evidence

- `out/rdna4-coverage-current/coverage.json` - Fixed before baseline for the 55
  custom-input blocker transition ledger.
- `out/rdna4-coverage-current/coverage-summary.json` - Current summary
  containing the 114 readiness blockers.
- `out/rdna4-coverage-current/blocker-ledger.json` - Current blocker ledger for
  cross-checking problem classifications.

### Codebase Maps

- `.planning/codebase/ARCHITECTURE.md` - Dataset, runtime, and reporting flow.
- `.planning/codebase/TESTING.md` - CPU-safe artifact and deterministic report
  testing patterns.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets

- `src/sol_execbench/core/dataset/readiness.py`: readiness classifier that
  currently emits custom-input blockers.
- `src/sol_execbench/core/dataset/profiler_timing_coverage.py`: coverage report
  model and totals used by RDNA4 coverage artifacts.
- `scripts/run_rdna4_profiler_timing_coverage.py`: current RDNA4 coverage
  report generation path.
- `out/rdna4-coverage-current/coverage.json`: current fixed baseline artifact.

### Established Patterns

- Coverage/report outputs should be deterministic and checksum-backed.
- Blocker classes should remain denominator-visible and must not be promoted to
  profiler-backed timing or passed validation without execution evidence.
- CPU-safe tests should validate report accounting and classification behavior
  without requiring RDNA4 hardware.

### Integration Points

- Phase 171 should consume Phase 170 custom-input support and recompute
  readiness/coverage.
- Transition ledgers should be machine-readable and useful to later Phase 174
  final closure.

</code_context>

<specifics>
## Specific Ideas

- Treat all 55 original custom-input readiness blockers as an explicit cohort.
- Report both the previous status and the new disposition for each problem.
- Preserve workload-level gaps honestly with `workload_transition_unavailable`.

</specifics>

<deferred>
## Deferred Ideas

- Quant readiness transitions are deferred to Phase 172.
- FlashInfer readiness transitions are deferred to Phase 173.
- Final all-114 readiness closure and public claim wording are deferred to
  Phase 174.

</deferred>

---

*Phase: 171-Custom Input Coverage Recompute*
*Context gathered: 2026-06-09*

