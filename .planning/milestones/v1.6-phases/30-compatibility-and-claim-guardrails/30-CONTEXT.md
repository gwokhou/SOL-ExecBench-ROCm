# Phase 30: Compatibility and Claim Guardrails - Context

**Gathered:** 2026-05-22
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 30 is the v1.6 compatibility and claim boundary gate. It proves that the
new analyzer, live timing, and derived scoring workflow did not break canonical
trace JSONL, public schemas, or primary `sol-execbench` CLI defaults, and that
CDNA3/NVIDIA equivalence claims remain explicitly out of scope.

</domain>

<decisions>
## Implementation Decisions

### Compatibility Gate Scope
- Verify trace JSONL fields/parsing, public schemas, primary `sol-execbench`
  help/defaults, and derived artifact separation.
- Treat the dataset runner `--amd-score-report` as an additive script option,
  not as a change to the primary `sol-execbench` CLI contract.
- Assert derived artifacts keep `canonical_output == trace_jsonl`,
  `derived == true`, and do not mutate `Trace` model dumps.
- Run existing public contract tests plus v1.6 focused tests.

### Claim Boundaries
- Documentation and tests should say v1.6 does not include real `gfx94*`
  full-suite validation.
- AMD-native derived reports must not claim NVIDIA B200, upstream SOLAR, or
  leaderboard equivalence.
- Bound and score artifacts must retain hardware model source, confidence, and
  validation status or evidence references.
- CDNA3, unsupported, and unvalidated warnings must remain active.

### Closure Strategy
- Prefer tests/docs and small stale-warning fixes over broad refactors.
- Update `CDNA3_NO_VALIDATION_WARNING` so it is not tied to v1.5.
- Run the relevant public-contract, AMD SOL, rocprof, AMD score, and dataset
  score tests.
- After Phase 30, continue autonomous lifecycle: audit, complete, cleanup.

### the agent's Discretion
All implementation details not fixed above are at the agent's discretion, with
trace/schema/primary CLI compatibility as the hard constraint.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `tests/sol_execbench/test_public_contract_guardrails.py` already covers
  solution/workload/trace/CLI compatibility and CDNA3 deferral docs.
- `tests/sol_execbench/test_amd_sol_bounds.py`,
  `tests/sol_execbench/test_rocm_profiler.py`, and
  `tests/sol_execbench/test_amd_native_score.py` already cover derived
  artifacts.
- `docs/analysis.md`, `docs/rocm_timing.md`, and `.planning/PROJECT.md` hold
  the current public claim wording.

### Established Patterns
- Derived artifacts expose `derived` and `canonical_output`.
- Claim guardrails are enforced with focused tests and explicit warning
  strings.
- Public CLI compatibility is tested through Click `CliRunner`.

### Integration Points
- Add v1.6-specific guardrail tests near existing public contract tests.
- Update stale CDNA3 warning text in `amd_score.py` and tests.
- Update docs if claim wording is missing or stale.

</code_context>

<specifics>
## Specific Ideas

Keep Phase 30 small and evidence-focused. The implementation phases already
added the core features; this phase should prove the milestone did not drift
from the user's hard compatibility constraint.

</specifics>

<deferred>
## Deferred Ideas

- Real CDNA3 `gfx94*` full-suite validation.
- Full test-suite execution if focused compatibility and v1.6 tests pass.
- Any cross-vendor leaderboard equivalence methodology.

</deferred>
