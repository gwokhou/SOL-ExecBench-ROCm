---
phase: 23
plan: 23-01
type: execute
wave: 1
depends_on: []
files_modified:
  - src/sol_execbench/core/bench/timing_policy.py
  - src/sol_execbench/core/bench/__init__.py
  - docs/user/rocm_timing.md
  - tests/sol_execbench/test_timing_policy.py
  - tests/sol_execbench/test_rocm_eval_timing_audit.py
autonomous: true
requirements:
  - TIME-01
  - TIME-02
  - TIME-03
  - TIME-04
---

# Phase 23: Timing Semantics and Policy - Plan

**Status:** Ready for execution  
**Created:** 2026-05-22  
**Requirements:** TIME-01, TIME-02, TIME-03, TIME-04

<objective>
Define the source classification and timing policy contract for HIP native,
Triton, PyTorch, and mixed/unknown timing sources before Phase 24 changes the
default timing implementation.
</objective>

<scope>
## In Scope

- Add a pure timing policy module under `src/sol_execbench/core/bench/`.
- Classify timing source types from existing public `SupportedLanguages` values.
- Define source-specific timer backend, activity-domain, aggregation, fallback,
  and interpretation metadata.
- Document the timing chimney model and PyTorch ROCm compatibility naming.
- Add focused tests for TIME-01 through TIME-04.

## Out of Scope

- Running `rocprofv3` or parsing profiler output.
- Replacing `time_runnable()` default timing behavior.
- Adding SOL bound or AMD-native scoring artifacts.
- Mutating canonical trace JSONL.
- Performing or claiming CDNA3 hardware validation.
</scope>

<threat_model>
| Threat | Severity | Mitigation |
|--------|----------|------------|
| T-23-01: A single policy hides different source timing semantics. | High | Expose source type, backend, activity domain, aggregation rule, and interpretation in every policy record. |
| T-23-02: Event timing fallback is mistaken for profiler-backed kernel timing. | High | Fallback policies must include `fallback_applied=True`, a reason, and explicit event-timing interpretation. |
| T-23-03: PyTorch ROCm `torch.cuda` or `ProfilerActivity.CUDA` names are misread as NVIDIA runtime dependencies. | Medium | Documentation and audit tests explain compatibility naming. |
| T-23-04: Phase 23 accidentally changes benchmark trace or execution semantics. | High | Keep work in pure policy/docs/tests; do not alter trace models or eval-driver timing calls. |
</threat_model>

<tasks>
<task id="23-01-01" type="execute">
  <title>Add timing policy model</title>
  <requirements>TIME-01, TIME-02, TIME-03</requirements>
  <files>
    <file>src/sol_execbench/core/bench/timing_policy.py</file>
    <file>src/sol_execbench/core/bench/__init__.py</file>
  </files>
  <read_first>
    <file>src/sol_execbench/core/diagnostics.py</file>
    <file>src/sol_execbench/core/data/solution.py</file>
    <file>src/sol_execbench/core/bench/timing.py</file>
    <file>.planning/phases/23-timing-semantics-and-policy/23-RESEARCH.md</file>
    <file>.planning/phases/23-timing-semantics-and-policy/23-PATTERNS.md</file>
  </read_first>
  <action>
    Create `src/sol_execbench/core/bench/timing_policy.py` with `str, Enum`
    types for `TimingSourceType`, `TimingBackend`, and `TimingActivityDomain`.
    Add a frozen policy record with fields for `source_type`, `backend`,
    `activity_domain`, `aggregation_rule`, `interpretation`,
    `fallback_applied`, and `reason`. Implement a classifier that maps
    `SupportedLanguages.PYTORCH` to `pytorch`, `SupportedLanguages.TRITON` to
    `triton`, and `SupportedLanguages.HIP_CPP`, `HIPBLAS`, `MIOPEN`, `CK`, and
    `ROCWMMA` to `hip_native`. Include an explicit `unknown` or equivalent
    source type for unsupported evidence. Export the module through
    `src/sol_execbench/core/bench/__init__.py` only if that package already
    exposes public helpers; otherwise keep imports direct in tests.
  </action>
  <acceptance_criteria>
    <criterion>`src/sol_execbench/core/bench/timing_policy.py` exists.</criterion>
    <criterion>The module defines enum values for `pytorch`, `triton`, `hip_native`, and `unknown` or equivalent source type.</criterion>
    <criterion>The classifier maps every current `SupportedLanguages` member to a timing source type or documented unsupported state.</criterion>
    <criterion>No code in this task invokes `rocprofv3`, `subprocess`, or GPU timing APIs.</criterion>
  </acceptance_criteria>
  <verify>
    <command>uv run pytest tests/sol_execbench/test_timing_policy.py</command>
  </verify>
</task>

<task id="23-01-02" type="execute">
  <title>Add source-specific policy table tests</title>
  <requirements>TIME-01, TIME-02, TIME-03</requirements>
  <files>
    <file>tests/sol_execbench/test_timing_policy.py</file>
  </files>
  <read_first>
    <file>tests/sol_execbench/test_rocm_diagnostics_reporting.py</file>
    <file>src/sol_execbench/core/bench/timing_policy.py</file>
    <file>src/sol_execbench/core/data/solution.py</file>
    <file>.planning/phases/23-timing-semantics-and-policy/23-VALIDATION.md</file>
  </read_first>
  <action>
    Add `tests/sol_execbench/test_timing_policy.py` using the direct assertion
    style from `test_rocm_diagnostics_reporting.py`. Cover all current
    `SupportedLanguages` mappings. Assert the policy table contains distinct
    policies for PyTorch, Triton, and HIP native sources, and that each policy
    exposes backend, activity domain, aggregation rule, interpretation,
    fallback flag, and reason. Add a test proving fallback/event timing is
    labeled as fallback and not reported as profiler-backed timing.
  </action>
  <acceptance_criteria>
    <criterion>`uv run pytest tests/sol_execbench/test_timing_policy.py` exits 0.</criterion>
    <criterion>Tests assert distinct policies for `pytorch`, `triton`, and `hip_native`.</criterion>
    <criterion>Tests fail if a policy lacks `interpretation` or `aggregation_rule`.</criterion>
    <criterion>Tests fail if fallback event timing has no fallback reason.</criterion>
  </acceptance_criteria>
  <verify>
    <command>uv run pytest tests/sol_execbench/test_timing_policy.py</command>
  </verify>
</task>

<task id="23-01-03" type="execute">
  <title>Document timing chimney semantics</title>
  <requirements>TIME-03, TIME-04</requirements>
  <files>
    <file>docs/user/rocm_timing.md</file>
    <file>tests/sol_execbench/test_rocm_eval_timing_audit.py</file>
  </files>
  <read_first>
    <file>docs/internal/analysis.md</file>
    <file>docs/user/ARCHITECTURE.md</file>
    <file>docs/user/solution.md</file>
    <file>tests/sol_execbench/test_rocm_eval_timing_audit.py</file>
    <file>.planning/phases/23-timing-semantics-and-policy/23-RESEARCH.md</file>
  </read_first>
  <action>
    Create `docs/user/rocm_timing.md` describing the accuracy-first timing rule and
    the source-specific chimney model. Include a table for `pytorch`, `triton`,
    `hip_native`, and fallback/unknown sources with columns for source type,
    timer backend, measured activity domain, aggregation rule, and
    interpretation. Explicitly state that PyTorch ROCm uses `torch.cuda` and
    CUDA-named profiler activity APIs as compatibility names. Extend
    `tests/sol_execbench/test_rocm_eval_timing_audit.py` with documentation
    assertions that `docs/user/rocm_timing.md` contains `kernel activity`, `HIP
    runtime`, `PyTorch operator attribution`, `fallback event timing`, and
    `source_type -> timer_backend -> interpretation`.
  </action>
  <acceptance_criteria>
    <criterion>`docs/user/rocm_timing.md` exists.</criterion>
    <criterion>`docs/user/rocm_timing.md` contains the text `source_type -> timer_backend -> interpretation`.</criterion>
    <criterion>`docs/user/rocm_timing.md` contains the strings `kernel activity`, `HIP runtime`, `PyTorch operator attribution`, and `fallback event timing`.</criterion>
    <criterion>`uv run pytest tests/sol_execbench/test_rocm_eval_timing_audit.py` exits 0.</criterion>
  </acceptance_criteria>
  <verify>
    <command>uv run pytest tests/sol_execbench/test_rocm_eval_timing_audit.py</command>
  </verify>
</task>

<task id="23-01-04" type="execute">
  <title>Run phase validation and compatibility guardrails</title>
  <requirements>TIME-01, TIME-02, TIME-03, TIME-04</requirements>
  <files>
    <file>tests/sol_execbench/test_timing_policy.py</file>
    <file>tests/sol_execbench/test_rocm_eval_timing_audit.py</file>
    <file>tests/sol_execbench/test_public_contract_guardrails.py</file>
  </files>
  <read_first>
    <file>.planning/phases/23-timing-semantics-and-policy/23-VALIDATION.md</file>
    <file>.planning/REQUIREMENTS.md</file>
    <file>.planning/ROADMAP.md</file>
  </read_first>
  <action>
    Run the Phase 23 quick and full validation commands. If failures are caused
    by the new policy/docs tests, fix the Phase 23 implementation. Do not
    weaken existing public-contract guardrails. Record any unavoidable deviation
    in the eventual phase summary during execution.
  </action>
  <acceptance_criteria>
    <criterion>`uv run pytest tests/sol_execbench/test_timing_policy.py tests/sol_execbench/test_rocm_eval_timing_audit.py` exits 0.</criterion>
    <criterion>`uv run pytest tests/sol_execbench/test_timing_policy.py tests/sol_execbench/test_rocm_eval_timing_audit.py tests/sol_execbench/test_public_contract_guardrails.py` exits 0.</criterion>
    <criterion>No canonical trace schema files are modified by Phase 23 execution.</criterion>
  </acceptance_criteria>
  <verify>
    <command>uv run pytest tests/sol_execbench/test_timing_policy.py tests/sol_execbench/test_rocm_eval_timing_audit.py tests/sol_execbench/test_public_contract_guardrails.py</command>
  </verify>
</task>
</tasks>

<verification>
## Verification

Run:

```bash
uv run pytest tests/sol_execbench/test_timing_policy.py tests/sol_execbench/test_rocm_eval_timing_audit.py
uv run pytest tests/sol_execbench/test_timing_policy.py tests/sol_execbench/test_rocm_eval_timing_audit.py tests/sol_execbench/test_public_contract_guardrails.py
uv run ruff check src/sol_execbench/core/bench/timing_policy.py tests/sol_execbench/test_timing_policy.py tests/sol_execbench/test_rocm_eval_timing_audit.py
```
</verification>

<success_criteria>
## Success Criteria

1. Maintainers can classify measured work as HIP native, Triton, PyTorch, or
   unknown/mixed before selecting a timing backend.
2. A timing policy table maps each source type to backend, activity domain,
   aggregation rule, and interpretation.
3. Source-specific timing chimneys are represented in code and documentation
   when a unified timing口径 would reduce accuracy.
4. Documentation explains kernel activity, HIP runtime/API activity, PyTorch
   operator attribution, and fallback event timing.
5. Phase 23 changes do not alter canonical trace JSONL or eval-driver timing
   execution behavior.
</success_criteria>

<must_haves>
## Must Haves

- TIME-01, TIME-02, TIME-03, and TIME-04 are all covered by tests or doc
  assertions.
- Policy models are pure and do not invoke profiler subprocesses.
- Fallback event timing is labeled as fallback with a reason.
- PyTorch ROCm compatibility naming is documented without implying NVIDIA
  runtime support.
</must_haves>
