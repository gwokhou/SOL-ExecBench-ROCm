# Phase 55: Ready Subset Selection And Bounded Execution Closure - Context

**Gathered:** 2026-05-23
**Status:** Ready for planning

<domain>
## Phase Boundary

This phase runs bounded ready subsets through the existing dataset execution
path and joins execution outputs back to readiness and evidence artifacts. It
may create temporary filtered workload files in output/staging paths, but must
not mutate canonical dataset files or canonical trace JSONL. Closure artifacts
describe attempted, skipped, failed, filtered, missing, and not-attempted states
without claiming full 235-problem validation, paper parity, or leaderboard
results.

</domain>

<decisions>
## Implementation Decisions

### Ready-Subset Execution Boundary
- Extend `scripts/run_dataset.py` to read Phase 54 `ready_subset.json` and
  select only referenced problem/workload pairs while preserving the existing
  `sol-execbench` subprocess path.
- Generate temporary filtered workload files only under output or execution
  staging paths. Never modify canonical dataset `workload.jsonl`.
- If no ready workloads are selected, generate a closure report with
  `no_ready_workloads` semantics and exit successfully.
- Apply filters by intersection: ready subset, category, limit, workload cap,
  and runner filters all contribute visible filtered/not-attempted reasons.

### Execution Controls And Provenance
- Reuse or extend existing `run_dataset.py` controls for category, limit,
  workload cap, timeout, warmup, iterations, rerun policy, and derived evidence
  flags.
- Closure provenance should record command args, dataset manifest checksum,
  readiness/ready-subset checksum, git commit, solution mode/name, benchmark
  config, timestamp, and output paths.
- `skipped_existing_pass` is a first-class closure state and should still
  generate or reference derived sidecars unless rerun policy requires execution.
- Per-workload/problem failures should not prevent the closure report from
  being written; report the failure state explicitly.

### Closure Report Shape
- Emit `execution_closure.json` and optionally a lightweight Markdown summary.
  Full parity gap reporting remains Phase 56.
- Use closure statuses `not_attempted`, `filtered`, `skipped_existing_pass`,
  `attempted_passed`, `attempted_failed`, `missing_trace`, and
  `derived_evidence_missing`.
- Join readiness, traces, summaries, logs, AMD score, AMD SOL v2, SOLAR
  derivation, and timing sidecar paths by problem ID plus workload UUID or
  row index.
- Read and reference canonical trace JSONL only; do not add closure fields or
  rewrite trace JSONL.

### Verification And Hardware Scope
- Automated tests should use fixtures and monkeypatched runner functions rather
  than real ROCm/GPU execution.
- If GPU hardware exists, real sample execution is optional manual validation,
  not a phase-pass requirement.
- Derived evidence checks should reuse existing Phase 52 sidecar generation and
  reference behavior.
- Claim guardrails must say bounded ready-subset closure is not full 235
  validation, paper parity, or leaderboard result; failures, blockers, and
  not-attempted states must stay visible.

### the agent's Discretion
The agent may choose exact helper names, closure schema fields, and Markdown
summary format as long as existing runner semantics are preserved, outputs are
deterministic, and canonical dataset/traces stay unchanged.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- Phase 54 added `ReadySubset` and readiness sidecars under
  `src/sol_execbench/core/dataset/`.
- `scripts/run_dataset.py` already handles problem discovery, category/limit,
  timeouts, rerun behavior, existing trace skips, derived AMD score reports,
  AMD SOL v2 sidecars, SOLAR derivation sidecars, and timing evidence paths.
- Canonical trace JSONL parsing uses `Trace` and existing report helpers.

### Established Patterns
- Keep the primary `sol-execbench` CLI unchanged for sidecar workflows.
- Put new sidecar metadata outside canonical trace JSONL.
- Use fixture tests and monkeypatch `run_cli` to verify runner behavior without
  GPU execution.

### Integration Points
- Extend `scripts/run_dataset.py` rather than adding a second runner.
- Add focused tests near `tests/sol_execbench/test_run_dataset_amd_score.py` or
  a new dataset closure test module.
- Closure claim guardrails belong in existing public contract guardrail tests
  and docs near `docs/analysis.md`.

</code_context>

<specifics>
## Specific Ideas

The core of this phase is visibility: the report should make skipped, filtered,
failed, missing trace, and derived-evidence-missing states explicit rather than
hiding them behind a single pass/fail count.

</specifics>

<deferred>
## Deferred Ideas

- Full parity gap report aggregation is deferred to Phase 56.
- Milestone release claim closure is deferred to Phase 57.
- Real full-suite hardware validation remains out of scope.

</deferred>
