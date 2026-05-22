---
phase: 24
plan: 24-01
type: execute
wave: 1
depends_on: []
files_modified:
  - src/sol_execbench/core/bench/rocm_profiler.py
  - docs/rocm_timing.md
  - tests/sol_execbench/test_rocm_profiler.py
  - tests/sol_execbench/test_rocm_eval_timing_audit.py
autonomous: true
requirements:
  - PROF-01
  - PROF-02
  - PROF-03
  - PROF-04
---

# Phase 24: rocprofv3 Default Timing Path - Plan

**Status:** Ready for execution  
**Created:** 2026-05-22  
**Requirements:** PROF-01, PROF-02, PROF-03, PROF-04

<objective>
Add a policy-aware `rocprofv3` wrapper/parser/evidence layer that can collect
and audit profiler-backed timing while preserving explicit fallback behavior.
</objective>

<tasks>
<task id="24-01-01" type="execute">
  <title>Add rocprofv3 command, parser, and evidence models</title>
  <requirements>PROF-01, PROF-04</requirements>
  <files>
    <file>src/sol_execbench/core/bench/rocm_profiler.py</file>
    <file>tests/sol_execbench/test_rocm_profiler.py</file>
  </files>
  <read_first>
    <file>src/sol_execbench/core/bench/timing_policy.py</file>
    <file>src/sol_execbench/core/reporting.py</file>
    <file>.planning/phases/24-rocprofv3-default-timing-path/24-RESEARCH.md</file>
  </read_first>
  <action>
    Create `rocm_profiler.py` with frozen dataclasses for parsed profiler rows
    and timing evidence. Add a `build_rocprofv3_command()` helper that returns
    a command containing `rocprofv3`, `--kernel-trace`, `--hip-runtime-trace`,
    `--output-format csv`, `--output-directory`, `--output-file`, `--`, and the
    application command. Add a robust CSV parser that normalizes headers,
    separates kernel activity rows from HIP runtime/API rows, and aggregates
    kernel duration in milliseconds.
  </action>
  <acceptance_criteria>
    <criterion>`src/sol_execbench/core/bench/rocm_profiler.py` exists.</criterion>
    <criterion>`tests/sol_execbench/test_rocm_profiler.py` proves command construction places app args after `--`.</criterion>
    <criterion>Fixture parser tests extract kernel rows and ignore HIP runtime rows for kernel duration.</criterion>
    <criterion>Evidence `to_dict()` includes tool version, gpu architecture, activity domain, aggregation rule, and parsed rows.</criterion>
  </acceptance_criteria>
  <verify>
    <command>uv run pytest tests/sol_execbench/test_rocm_profiler.py</command>
  </verify>
</task>

<task id="24-01-02" type="execute">
  <title>Add policy-aware default and fallback selection</title>
  <requirements>PROF-02, PROF-03</requirements>
  <files>
    <file>src/sol_execbench/core/bench/rocm_profiler.py</file>
    <file>tests/sol_execbench/test_rocm_profiler.py</file>
  </files>
  <read_first>
    <file>src/sol_execbench/core/bench/timing_policy.py</file>
    <file>src/sol_execbench/core/diagnostics.py</file>
  </read_first>
  <action>
    Add a helper that accepts a Phase 23 `TimingPolicy` and a `rocprofv3`
    availability flag. It should choose profiler-backed timing when the policy
    backend is `rocprofv3` and `rocprofv3` is available. Otherwise it returns
    explicit fallback evidence or fallback policy with backend, reason, and
    interpretation. Do not silently substitute event timing.
  </action>
  <acceptance_criteria>
    <criterion>Tests prove HIP native and Triton policies select profiler-backed timing when `rocprofv3_available=True`.</criterion>
    <criterion>Tests prove unavailable `rocprofv3` produces explicit fallback metadata.</criterion>
    <criterion>Tests prove PyTorch attribution-only policy does not masquerade as `rocprofv3` kernel activity.</criterion>
  </acceptance_criteria>
  <verify>
    <command>uv run pytest tests/sol_execbench/test_rocm_profiler.py</command>
  </verify>
</task>

<task id="24-01-03" type="execute">
  <title>Document profiler evidence semantics</title>
  <requirements>PROF-03, PROF-04</requirements>
  <files>
    <file>docs/rocm_timing.md</file>
    <file>tests/sol_execbench/test_rocm_eval_timing_audit.py</file>
  </files>
  <read_first>
    <file>docs/rocm_timing.md</file>
    <file>tests/sol_execbench/test_rocm_eval_timing_audit.py</file>
  </read_first>
  <action>
    Extend `docs/rocm_timing.md` with a `Profiler Evidence` section naming
    required fields: tool version, GPU architecture, activity domain,
    aggregation rule, parsed timing rows, backend, fallback reason, and
    interpretation. Extend the audit test to assert those strings exist.
  </action>
  <acceptance_criteria>
    <criterion>`docs/rocm_timing.md` contains `Profiler Evidence`.</criterion>
    <criterion>`uv run pytest tests/sol_execbench/test_rocm_eval_timing_audit.py` exits 0.</criterion>
  </acceptance_criteria>
  <verify>
    <command>uv run pytest tests/sol_execbench/test_rocm_eval_timing_audit.py</command>
  </verify>
</task>

<task id="24-01-04" type="execute">
  <title>Run Phase 24 verification</title>
  <requirements>PROF-01, PROF-02, PROF-03, PROF-04</requirements>
  <files>
    <file>tests/sol_execbench/test_rocm_profiler.py</file>
    <file>tests/sol_execbench/test_timing_policy.py</file>
    <file>tests/sol_execbench/test_rocm_eval_timing_audit.py</file>
    <file>tests/sol_execbench/test_public_contract_guardrails.py</file>
  </files>
  <read_first>
    <file>.planning/phases/24-rocprofv3-default-timing-path/24-VALIDATION.md</file>
  </read_first>
  <action>
    Run quick tests, full tests, and ruff for the new profiler module and tests.
  </action>
  <acceptance_criteria>
    <criterion>Quick and full pytest commands from `24-VALIDATION.md` exit 0.</criterion>
    <criterion>Ruff exits 0 for modified Python files.</criterion>
  </acceptance_criteria>
  <verify>
    <command>uv run pytest tests/sol_execbench/test_rocm_profiler.py tests/sol_execbench/test_timing_policy.py tests/sol_execbench/test_rocm_eval_timing_audit.py tests/sol_execbench/test_public_contract_guardrails.py</command>
  </verify>
</task>
</tasks>

<verification>
```bash
uv run pytest tests/sol_execbench/test_rocm_profiler.py tests/sol_execbench/test_rocm_eval_timing_audit.py
uv run pytest tests/sol_execbench/test_rocm_profiler.py tests/sol_execbench/test_timing_policy.py tests/sol_execbench/test_rocm_eval_timing_audit.py tests/sol_execbench/test_public_contract_guardrails.py
uv run ruff check src/sol_execbench/core/bench/rocm_profiler.py tests/sol_execbench/test_rocm_profiler.py tests/sol_execbench/test_rocm_eval_timing_audit.py
```
</verification>
