# Phase 24: rocprofv3 Default Timing Path - Context

**Gathered:** 2026-05-22
**Status:** Ready for planning
**Mode:** Autonomous smart discuss, infrastructure path

<domain>
## Phase Boundary

Implement profiler-backed ROCm timing and make it the default timing path when
the Phase 23 policy says it is the most accurate supported backend for the
classified source type.

</domain>

<decisions>
## Implementation Decisions

### Timing Accuracy
- Timing accuracy remains the highest rule.
- `rocprofv3` kernel activity should be preferred only when it can be collected,
  parsed, and interpreted correctly.
- Fallback timing is acceptable only when explicitly labeled with backend,
  reason, and interpretation.

### Evidence and Outputs
- Profiler timing evidence must be a derived artifact and must not mutate
  canonical trace JSONL.
- Evidence must include tool version, GPU architecture, activity domain,
  aggregation rule, and parsed timing rows sufficient to audit duration.
- Keep profiler output in controlled evidence directories.

### Source Semantics
- Phase 24 consumes the Phase 23 timing policy contract.
- Do not collapse PyTorch, Triton, and HIP native semantics into one unlabeled
  duration if doing so would reduce accuracy.
- Do not include Triton JIT/autotune or PyTorch setup overhead in steady-state
  device timing unless evidence explicitly labels that interpretation.

### the agent's Discretion
- Exact wrapper/parser module names.
- Exact fixture format for representative `rocprofv3` outputs.
- Whether default dispatch is wired fully into `time_runnable()` in this phase
  or introduced behind a policy-aware helper first, as long as PROF-02 is met.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/sol_execbench/core/bench/timing.py` owns current timing execution.
- `src/sol_execbench/core/bench/timing_policy.py` defines source/backend/domain
  policy semantics from Phase 23.
- `src/sol_execbench/core/reporting.py` shows derived evidence patterns that do
  not mutate trace JSONL.
- `src/sol_execbench/core/diagnostics.py` has backend/fallback vocabulary and
  ROCm tool detection helpers.

### Established Patterns
- Internal helpers use `str, Enum` and frozen dataclasses for policy-like data.
- Public trace schema compatibility is protected by tests.
- ROCm compatibility names such as `torch.cuda` are documented rather than
  removed when PyTorch ROCm requires them.

### Integration Points
- `time_runnable()` is the current eval-driver timing boundary.
- Phase 24 should add parser/wrapper tests before relying on local hardware
  traces.
- `docs/rocm_timing.md` is the user-facing timing semantics document.

</code_context>

<specifics>
## Specific Ideas

- Start with `rocprofv3` fixture parsing and evidence models.
- Use Phase 23 policy results to decide profiler-backed versus fallback timing.
- Preserve reward-hack and public-contract guardrails.

</specifics>

<deferred>
## Deferred Ideas

- Real CDNA3 `gfx94*` full-suite validation remains out of scope.
- AMD SOL bound and scoring integration remain Phase 25 and Phase 26 scope.

</deferred>
