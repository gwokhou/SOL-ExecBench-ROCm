---
phase: 26
plan: 26-01
type: execute
wave: 1
depends_on:
  - phase-25
files_modified:
  - src/sol_execbench/core/scoring/amd_score.py
  - src/sol_execbench/core/scoring/__init__.py
  - tests/sol_execbench/test_amd_native_score.py
  - docs/analysis.md
autonomous: true
requirements:
  - SCORE-01
  - SCORE-02
  - SCORE-03
  - SCORE-04
  - COMPAT-01
  - COMPAT-02
  - CLAIM-01
  - CLAIM-02
---

# Phase 26: AMD-native Scoring and Guarded Reports - Plan

**Status:** Complete 2026-05-22  
**Created:** 2026-05-22  
**Requirements:** SCORE-01, SCORE-02, SCORE-03, SCORE-04, COMPAT-01,
COMPAT-02, CLAIM-01, CLAIM-02

<objective>
Generate derived AMD-native scoring reports from measured timing and AMD SOL
bound artifacts, while preserving existing public benchmark contracts and
explicitly guarding unvalidated CDNA3 claims.
</objective>

<tasks>
<task id="26-01-01" type="execute">
  <requirements>SCORE-01, SCORE-03, SCORE-04, COMPAT-02, CLAIM-01, CLAIM-02</requirements>
  <files>
    <file>src/sol_execbench/core/scoring/amd_score.py</file>
    <file>src/sol_execbench/core/scoring/__init__.py</file>
    <file>tests/sol_execbench/test_amd_native_score.py</file>
  </files>
  <read_first>
    <file>src/sol_execbench/sol_score.py</file>
    <file>src/sol_execbench/core/scoring/amd_sol.py</file>
    <file>src/sol_execbench/core/reporting.py</file>
    <file>src/sol_execbench/core/scoring_guardrails.py</file>
  </read_first>
  <action>
    Add a derived AMD-native score report module. Compute per-workload scores
    with the existing `sol_score()` formula using measured latency, baseline
    latency, and the aggregate AMD SOL bound. Preserve evidence references and
    guardrail warnings for unsupported ops, incomplete inputs, unvalidated
    hardware, and CDNA3 no-validation status. Aggregate suite mean scores
    without mutating canonical traces.
  </action>
  <acceptance_criteria>
    <criterion>Per-problem reports include measured latency, baseline latency, SOL bound, score, claim level, warnings, and evidence refs.</criterion>
    <criterion>Suite reports include schema version, `derived=True`, `canonical_output=trace_jsonl`, mean score, and per-workload entries.</criterion>
    <criterion>CDNA3 unvalidated model emits explicit no-validation guardrail.</criterion>
  </acceptance_criteria>
  <verify>
    <command>uv run pytest tests/sol_execbench/test_amd_native_score.py</command>
  </verify>
</task>

<task id="26-01-02" type="execute">
  <requirements>SCORE-02, COMPAT-01, COMPAT-02</requirements>
  <files>
    <file>docs/analysis.md</file>
    <file>tests/sol_execbench/test_amd_native_score.py</file>
  </files>
  <action>
    Document AMD-native score reports as derived artifacts and state that
    baseline comparison remains baseline-relative with no NVIDIA B200, SOLAR,
    or leaderboard equivalence claim.
  </action>
  <acceptance_criteria>
    <criterion>Docs state AMD-native score reports are derived artifacts.</criterion>
    <criterion>Public contract and baseline comparison tests continue to pass.</criterion>
  </acceptance_criteria>
</task>
</tasks>

<verification>
```bash
uv run pytest tests/sol_execbench/test_amd_native_score.py tests/sol_execbench/test_amd_sol_bounds.py tests/sol_execbench/test_baseline_comparison.py tests/sol_execbench/test_trace_reporting_and_score_guardrails.py tests/sol_execbench/test_public_contract_guardrails.py
uv run ruff check src/sol_execbench/core/scoring/amd_score.py tests/sol_execbench/test_amd_native_score.py
```
</verification>
