# Phase 28: Live rocprofv3 Timing Integration - Context

**Gathered:** 2026-05-22
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 28 adds a reusable live `rocprofv3` timing collection adapter for
benchmark or dataset execution. The adapter must preserve source-specific
timing semantics, produce derived timing evidence, and avoid changing canonical
trace JSONL or primary CLI defaults.

</domain>

<decisions>
## Implementation Decisions

### Live Integration Boundary
- Provide a reusable execution adapter/helper that benchmark or dataset paths
  can call; do not change the primary `sol-execbench` default behavior in this
  phase.
- Use CSV as the first supported profiler output format; keep rocpd/json as
  future extensions.
- Require caller-provided controlled evidence directories and deterministic
  output file prefixes.
- If live collection cannot provide profiler evidence, return explicitly
  labeled fallback evidence or selection metadata rather than silently treating
  event timing as profiler timing.

### Timing Semantics
- HIP native and Triton timing use `rocprofv3` kernel activity rows with
  post-warmup aggregation semantics.
- PyTorch keeps operator attribution semantics and must not masquerade as
  raw `rocprofv3` kernel activity.
- Mixed and unknown sources use explicit fallback or unsupported evidence with
  reasons instead of automatic guessing.
- Compile, autotune, warmup, unrelated kernel rows, and event fallback must be
  excluded or explicitly labeled.

### Test and Contract Boundary
- Test command building, subprocess invocation through mocks, CSV parsing, and
  fallback labels.
- Do not require real `rocprofv3` or GPU hardware in unit tests.
- Do not modify canonical trace models or trace JSONL.
- Update `docs/user/rocm_timing.md` to describe the live adapter and chimney-style
  source-specific timing outputs.

### the agent's Discretion
All implementation details not fixed above are at the agent's discretion,
provided timing accuracy and public contract preservation remain the highest
priority.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/sol_execbench/core/bench/rocm_profiler.py` already builds `rocprofv3`
  commands, parses representative CSV output, builds timing evidence, and
  selects fallback timing metadata.
- `src/sol_execbench/core/bench/timing_policy.py` already models
  source-specific timer backend, activity domain, aggregation rule, and
  interpretation.
- `tests/sol_execbench/test_rocm_profiler.py` is the focused test file for this
  phase.

### Established Patterns
- Timing and profiler evidence are derived methodology artifacts with
  `to_dict()` payloads.
- Fallback timing is explicit and labeled.
- PyTorch ROCm CUDA-named APIs are documented as compatibility naming, not
  NVIDIA runtime evidence.

### Integration Points
- Add live collection helpers to `rocm_profiler.py`.
- Keep event timing in `timing.py` unchanged for the default path.
- Update `docs/user/rocm_timing.md`.

</code_context>

<specifics>
## Specific Ideas

Expose source-type to timer-type semantics as a chimney rather than hiding
HIP, Triton, and PyTorch behind one timer口径.

</specifics>

<deferred>
## Deferred Ideas

- Making primary `sol-execbench` default always execute under `rocprofv3`.
- Real hardware profiler validation as a required unit-test path.
- rocpd/json profiler output parsing.

</deferred>
