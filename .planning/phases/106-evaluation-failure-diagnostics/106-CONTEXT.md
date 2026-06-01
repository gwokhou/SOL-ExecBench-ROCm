# Phase 106: Evaluation Failure Diagnostics - Context

**Gathered:** 2026-06-01
**Status:** Ready for planning

<domain>
## Phase Boundary

This phase improves single-evaluation failure diagnostics when the evaluator
produces no parseable trace JSONL. It persists bounded stdout/stderr evidence
for no-trace and nonzero failures, and points CLI users to that evidence
without changing canonical trace JSONL or public trace schemas.

</domain>

<decisions>
## Implementation Decisions

### Diagnostic Boundary
- Persist diagnostics as trace-adjacent or staging-adjacent sidecar JSON, not as
  new trace fields.
- Keep canonical Trace, Definition, Workload, Solution, correctness, timing,
  score, and evaluator contract schemas unchanged.
- Bound captured stdout and stderr tails so noisy libraries cannot create
  unbounded artifacts.
- Diagnostics should be written for no-trace outcomes even when the evaluator
  exits zero with non-JSON stdout.

### CLI Behavior
- CLI failure messages should include the diagnostic sidecar path when one is
  written.
- `--verbose` may still print raw stderr, but diagnostics must not require
  verbose mode.
- `--keep-staging` remains useful, but users should not need to inspect staging
  manually to find the first useful failure context.
- Normal passing trace output and JSON output behavior must remain unchanged.

### Testing Boundary
- Add focused tests for helper behavior rather than requiring live ROCm/GPU.
- Cover non-JSON stdout, library noise, nonzero exits, empty trace output, and
  bounded tail behavior.
- Guard that no diagnostic fields are added to canonical trace payloads.
- Avoid expanding scope into dataset-runner closure behavior; that is queued
  for v1.24.

### the agent's Discretion
The implementation may choose the exact helper names, sidecar schema shape, and
path derivation as long as the sidecar is deterministic, bounded, and clearly
diagnostic-only.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/sol_execbench/cli/main.py` already writes optional environment,
  profiler, and static-evidence sidecars beside `output_file` when available.
- `ProblemPackager.convert_stdout_to_traces()` in
  `src/sol_execbench/driver/problem_packager.py` skips non-JSON lines and
  returns parsed `Trace` objects.
- Existing profiler/static evidence code uses bounded `stdout_tail` and
  `stderr_tail` patterns.

### Established Patterns
- Optional evidence is diagnostic sidecar metadata and must not mutate
  canonical trace JSONL.
- CLI status messages are printed through the Rich console on stderr.
- Tests import private CLI helpers directly for focused sidecar behavior.

### Integration Points
- The no-trace branch in `src/sol_execbench/cli/main.py` currently prints
  `No traces produced`, optionally prints stderr, closes the packager, and
  exits with status 1.
- The hard failure branch where the process exits nonzero with empty stdout
  currently prints `Evaluation failed`, optionally prints stderr, closes the
  packager, and exits with status 1.
- `tests/sol_execbench/test_cli_environment_snapshot.py` is the nearest
  existing test home for CLI sidecar helpers.

</code_context>

<specifics>
## Specific Ideas

Use the recommended boundary accepted by the user: sidecar diagnostics only,
bounded stdout/stderr tails, CLI points to sidecar, canonical trace JSONL stays
unchanged.

</specifics>

<deferred>
## Deferred Ideas

- Inline trace/result schema diagnostics are out of scope.
- Dataset-scale no-trace closure behavior is queued for v1.24.
- Complete hard sandboxing remains a future standalone milestone.

</deferred>
