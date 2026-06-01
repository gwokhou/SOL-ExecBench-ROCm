# Phase 95: Eval Driver Runtime Decomposition - Context

**Gathered:** 2026-06-01
**Status:** Ready for planning
**Mode:** Auto-generated (infrastructure/refactor phase)

<domain>
## Phase Boundary

This phase continues thinning `src/sol_execbench/driver/templates/eval_driver.py`
by moving deterministic runtime helper behavior into importable package modules
with focused unit tests. The generated template remains the staging-directory
subprocess shell responsible for stdout redirection, dynamic imports, workload
loop orchestration, and JSONL trace emission.

</domain>

<decisions>
## Implementation Decisions

### the agent's Discretion
- Extract only cohesive helpers that can be unit-tested without a full staged
  subprocess.
- Preserve status priority, trace schema, reward-hack checks, clock-lock
  messages, and existing driver smoke behavior.
- Do not move the full workload loop in this phase.
- Prefer package code under `src/sol_execbench/core/bench/` because the helpers
  are evaluation-runtime primitives used by the template.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/sol_execbench/core/bench/utils.py` already owns `make_eval()` and the
  extracted `call_and_collect_outputs()` helper.
- `src/sol_execbench/core/bench/correctness.py` owns structural output
  validation and numerical correctness helpers.
- `src/sol_execbench/core/bench/reward_hack.py` owns static source review and
  runtime integrity checks.

### Established Patterns
- The eval driver imports package helpers after stdout redirection and staging
  directory setup.
- Template integration is validated by
  `tests/sol_execbench/driver/test_eval_driver.py`.
- Helper-level behavior should be tested under `tests/sol_execbench/core/bench/`.

### Integration Points
- `src/sol_execbench/driver/templates/eval_driver.py` currently owns staged
  problem loading, reference module loading, native ROCm module loading,
  Python user module import, and blocking dynamic `torch.utils.cpp_extension`
  load calls.
- `src/sol_execbench/core/data/solution.py` defines `SupportedLanguages` and
  solution specs used to resolve native versus Python import paths.

</code_context>

<specifics>
## Specific Ideas

Start with staged problem/reference/user function loading and dynamic extension
blocking helpers. These are deterministic enough for CPU unit tests and remove
template-only behavior without changing the workload loop.

</specifics>

<deferred>
## Deferred Ideas

- Moving the full correctness/timing workload loop.
- Changing reward-hack status semantics or adding hard sandbox behavior.

</deferred>
