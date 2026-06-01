# Phase 109: Eval Driver Responsibility Boundaries - Context

**Gathered:** 2026-06-01
**Status:** Ready for planning

<domain>
## Phase Boundary

This phase narrows generated eval-driver responsibilities by moving additional
trace emission and reward-hack check behavior into importable helpers while
preserving staged subprocess semantics and public trace contracts.

</domain>

<decisions>
## Implementation Decisions

### Scope
- Make a bounded extraction rather than a full eval-driver rewrite.
- Prioritize helper seams that can be tested without staging the full driver.
- Preserve canonical trace JSONL, correctness, timing, score, and evaluator
  contract schemas.
- Preserve generated driver subprocess wiring and staging setup.

### Helper Boundaries
- Move strict JSONL trace emission into `core/bench/eval_runtime.py`.
- Move reward-hack check exception routing into an importable helper that
  returns a diagnostic message instead of emitting traces directly.
- Keep driver-local code responsible for assembling workload-specific traces and
  calling `_emit()`.

### Integrity Boundary
- Ensure helper names used by the generated driver remain included in the
  integrity snapshot if user code can reach and monkey-patch them.
- Keep tests focused on helper behavior plus existing eval-driver integration
  tests.

### the agent's Discretion
The exact function names and amount of template simplification are flexible as
long as the phase creates real importable seams and does not weaken benchmark
semantics.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/sol_execbench/core/bench/eval_runtime.py` already owns staged problem
  loading, reference loading, user loading, and timing helper seams.
- `eval_driver.py` currently has a local `_emit()` helper and local
  `_reward_hack_check()` wrapper.
- Existing eval-driver tests cover strict JSON emission, reward-hack traces,
  noisy stdout, and subprocess behavior.

### Established Patterns
- Helper extraction should preserve generated template behavior and keep
  template code as staged orchestration glue.
- Trace contract changes require explicit schema guardrails and are out of
  scope here.

### Integration Points
- The generated driver imports helper names from `eval_runtime.py`.
- `_CRITICAL_NAMES` protects benchmark-critical helpers from monkey-patching.

</code_context>

<specifics>
## Specific Ideas

Extract `emit_trace_jsonl()` and `run_reward_hack_check()` into
`eval_runtime.py`, update the template to use them, add unit tests, and keep the
existing driver subprocess tests passing.

</specifics>

<deferred>
## Deferred Ideas

- Full correctness/timing loop decomposition remains future work.
- No trace schema changes.

</deferred>
