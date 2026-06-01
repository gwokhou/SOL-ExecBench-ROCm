# Phase 101: Eval Driver Diagnostics And Framing - Context

**Gathered:** 2026-06-01
**Status:** Ready for planning

<domain>
## Phase Boundary
Maintainers can diagnose reference timing and output-framing behavior through importable helpers while preserving staged evaluator semantics. Scope is limited to EVAL-01..04: reference timing helper extraction, explicit reference timing failure semantics, stdout/stderr JSONL framing regressions, and moving avoidable pure logic out of `src/sol_execbench/driver/templates/eval_driver.py`.
</domain>

<decisions>
## Implementation Decisions

### the agent's Discretion
All implementation choices are at the agent's discretion - pure infrastructure and diagnostics phase. Preserve Trace, Evaluation, Performance, correctness, timing, and JSONL public schemas unless a narrowly additive diagnostic field is already supported by existing models. Prefer helpers under `src/sol_execbench/core/bench/eval_runtime.py` or a nearby bench module so behavior can be unit-tested without staging a full generated driver.
</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/sol_execbench/driver/templates/eval_driver.py` owns generated evaluator orchestration and currently swallows reference timing failures by leaving `_ref_latency_ms = 0.0`.
- `src/sol_execbench/core/bench/eval_runtime.py` already contains importable staging/runtime helpers for loading problems, reference functions, and user functions.
- `src/sol_execbench/core/bench/utils.py` owns `make_eval` and `call_and_collect_outputs`.
- Existing tests include `tests/sol_execbench/driver/test_eval_driver.py` and `tests/sol_execbench/core/bench/test_eval_runtime.py`.

### Established Patterns
- Keep the generated eval driver self-contained but delegate pure helper behavior to `sol_execbench.core.bench.*`.
- Keep trace JSONL strict and only emitted through the real stdout; noisy user/library output belongs on stderr.
- Add focused unit tests for helper behavior plus driver regression tests where framing or staged semantics matter.

### Integration Points
- `src/sol_execbench/driver/templates/eval_driver.py`
- `src/sol_execbench/core/bench/eval_runtime.py`
- `tests/sol_execbench/driver/test_eval_driver.py`
- `tests/sol_execbench/core/bench/test_eval_runtime.py`
</code_context>

<specifics>
## Specific Ideas
- Extract a reference timing helper that returns both latency and failure detail/status rather than requiring the generated driver to inline try/except/pass logic.
- Ensure requested reference timing failures are visible in evaluation log/status semantics while preserving benchmark correctness status where appropriate.
- Add tests proving stdout redirection and `_emit` framing still produce parseable JSONL when imports or user code print noise.
</specifics>

<deferred>
## Deferred Ideas
- New public Trace schema fields unless unavoidable and explicitly guarded.
- Changes to correctness thresholds, timing backend policy, or reward-hack scope beyond reference timing diagnostics.
- Hard sandboxing or adversarial output isolation beyond current stdout/stderr framing.
</deferred>
