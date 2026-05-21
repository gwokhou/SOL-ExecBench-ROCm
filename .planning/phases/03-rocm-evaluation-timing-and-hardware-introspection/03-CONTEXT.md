# Phase 3: ROCm Evaluation, Timing, and Hardware Introspection - Context

**Gathered:** 2026-05-21
**Status:** Ready for planning

<domain>
## Phase Boundary

Port the isolated evaluation runtime and benchmark-integrity mechanisms to ROCm.
This phase owns `eval_driver.py`, benchmark helper behavior used by evaluation,
ROCm timing/integrity checks, environment and hardware reporting, and direct
tests for trace JSONL discipline under PyTorch ROCm, Triton ROCm, and HIP/C++.

This phase does not migrate public examples, replace library-specific examples,
overhaul the full test-suite marker taxonomy, or prove RDNA 4/CDNA 3 matrix
coverage; those remain Phase 4 and Phase 5 work.

</domain>

<decisions>
## Implementation Decisions

### Evaluation Runtime Boundary
- Make PyTorch ROCm, Triton ROCm, and HIP/C++ shared-object paths runnable through `eval_driver.py`.
- Replace executable CUDA assumptions in evaluator paths, while leaving broader examples and docs to later phases.
- Preserve trace JSON discipline: build/runtime library noise belongs in logs or stderr; stdout remains valid trace JSONL.
- Preserve public trace fields and benchmark behavior unless ROCm requires a specific, documented change.

### Timing And Integrity
- Replace CUPTI/CUDA timing with ROCm/PyTorch timing primitives where available, using clear abstraction boundaries for Phase 5 hardware validation.
- Preserve the anti-cheat intent of asynchronous-work hiding tests, rewritten around ROCm-visible synchronization behavior.
- If precise ROCm timing support is unavailable, fail with actionable diagnostics rather than silently falling back to wall-clock timing.
- Require successful clock locking before benchmarks run.

### Hardware Introspection
- Prefer Python/PyTorch HIP information plus `rocminfo` and `rocm-smi` probes with graceful actionable errors.
- Report concrete `gfx...` targets plus ROCm/HIP versions, not only family labels.
- Fail only when a missing ROCm tool is required for benchmark integrity; otherwise report unavailable with remediation.
- Preserve any current device-selection shape if present; keep multi-GPU policy minimal until hardware validation.

### Verification Scope
- Prioritize unit and focused integration tests for eval driver behavior, timing abstraction, trace discipline, and hardware reports.
- Require local ROCm host smoke checks where tools exist; leave RDNA 4/CDNA 3 matrix proof to Phase 5.
- On non-ROCm CI/dev machines, skip or xfail hardware-only tests with explicit markers while keeping pure logic tests active.
- Defer example/library migration, full suite marker overhaul, and cross-architecture certification.

### the agent's Discretion
The planner may choose exact helper/module boundaries, subprocess parsing details,
and pytest marker names as long as the decisions above are preserved.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/sol_execbench/driver/templates/eval_driver.py` is the isolated runtime script that imports user code, generates inputs, evaluates correctness, times performance, and emits trace JSONL.
- `src/sol_execbench/core/bench/` owns input generation, output normalization, correctness checks, timing, clock locking, environment snapshots, and reward-hack detection.
- `src/sol_execbench/driver/problem_packager.py` stages validated problem files and now compiles ROCm native solutions through the Phase 2 HIP/C++ schema/build path.
- `src/sol_execbench/core/data/trace.py` and related data models define the trace JSON contract.

### Established Patterns
- Evaluation runs in a subprocess rather than the CLI process.
- The eval driver redirects library noise away from stdout so stdout can remain parseable JSONL.
- Native solutions compile before execution and load through the `benchmark_kernel.so` contract.
- Unit tests should mock hardware/tool probes where possible and reserve real hardware smoke for marker-gated tests.

### Integration Points
- CLI execution depends on `ProblemPackager.compile()` and `ProblemPackager.execute()` command contracts.
- `eval_driver.py` reads staged `definition.json`, `workload.jsonl`, `solution.json`, `config.json`, and optional `benchmark_kernel.so`.
- Phase 2 changed native schema/build names to `hip_cpp`, `gfx1200`, and `hip_cflags`; Phase 3 should build on those names rather than reintroducing CUDA aliases.
- Environment and timing reports must remain compatible with existing trace parsing and score utilities.

</code_context>

<specifics>
## Specific Ideas

- Require clock-lock success before benchmark execution, per user override during smart discuss.
- Use `uv run --no-sync` for focused tests when full dependency synchronization would unnecessarily download large ROCm wheels.
- Keep CUDA/NVIDIA cleanup scoped to Phase 3 evaluator/timing/hardware paths; examples and docs are later phases.

</specifics>

<deferred>
## Deferred Ideas

- Public example and library migration is deferred to Phase 4.
- Full RDNA 4 and CDNA 3 validation, broad marker taxonomy, and cross-architecture certification are deferred to Phase 5.
- User-facing documentation and compliance updates are deferred to Phase 6.

</deferred>
