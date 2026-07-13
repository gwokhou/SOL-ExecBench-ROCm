---
phase: 25
plan: 25-01
type: execute
wave: 1
depends_on: []
files_modified:
  - src/sol_execbench/core/scoring/amd_sol.py
  - src/sol_execbench/core/scoring/__init__.py
  - docs/internal/analysis.md
  - tests/sol_execbench/test_amd_sol_bounds.py
autonomous: true
requirements:
  - SOL-01
  - SOL-02
  - SOL-03
  - SOL-04
---

# Phase 25: AMD SOL Bound Foundation - Plan

**Status:** Complete 2026-05-22  
**Created:** 2026-05-22  
**Requirements:** SOL-01, SOL-02, SOL-03, SOL-04

<objective>
Build a conservative AMD SOL bound foundation with graph extraction, FLOP/byte
analysis, hardware model entries, and auditable bound artifacts.
</objective>

<tasks>
<task id="25-01-01" type="execute">
  <requirements>SOL-01, SOL-02, SOL-03, SOL-04</requirements>
  <files>
    <file>src/sol_execbench/core/scoring/amd_sol.py</file>
    <file>tests/sol_execbench/test_amd_sol_bounds.py</file>
  </files>
  <read_first>
    <file>src/sol_execbench/core/data/definition.py</file>
    <file>src/sol_execbench/core/reporting.py</file>
    <file>.planning/phases/25-amd-sol-bound-foundation/25-RESEARCH.md</file>
  </read_first>
  <action>
    Implement `amd_sol.py` with graph node, work estimate, hardware model, op
    bound, and aggregate artifact dataclasses. Extract matmul and elementwise
    nodes from `Definition.reference`. Estimate bytes from resolved
    input/output tensor shapes and dtype sizes. Estimate matmul FLOPs for
    supported patterns and mark unsupported/inexact estimates with confidence
    and rationale. Add `default_amd_hardware_models()` with RDNA4 and
    unvalidated CDNA3 entries. Add `build_amd_sol_bound_artifact()`.
  </action>
  <acceptance_criteria>
    <criterion>`uv run pytest tests/sol_execbench/test_amd_sol_bounds.py` exits 0.</criterion>
    <criterion>Artifact `to_dict()` includes graph nodes, work estimates, hardware model, per-op bounds, aggregate bound, and schema version.</criterion>
    <criterion>CDNA3 hardware model has validation status `unvalidated`.</criterion>
  </acceptance_criteria>
  <verify>
    <command>uv run pytest tests/sol_execbench/test_amd_sol_bounds.py</command>
  </verify>
</task>

<task id="25-01-02" type="execute">
  <requirements>SOL-04</requirements>
  <files>
    <file>docs/internal/analysis.md</file>
  </files>
  <read_first>
    <file>docs/internal/analysis.md</file>
  </read_first>
  <action>
    Extend the AMD-native score interpretation section to state that AMD-native
    scoring requires an AMD SOL bound artifact with graph nodes, FLOP/byte
    evidence, hardware model source/confidence, and validation status.
  </action>
  <acceptance_criteria>
    <criterion>`docs/internal/analysis.md` contains `AMD SOL bound artifact`.</criterion>
  </acceptance_criteria>
  <verify>
    <command>uv run pytest tests/sol_execbench/test_amd_sol_bounds.py</command>
  </verify>
</task>
</tasks>

<verification>
```bash
uv run pytest tests/sol_execbench/test_amd_sol_bounds.py tests/sol_execbench/test_trace_reporting_and_score_guardrails.py tests/sol_execbench/test_public_contract_guardrails.py
uv run ruff check src/sol_execbench/core/scoring/amd_sol.py tests/sol_execbench/test_amd_sol_bounds.py
```
</verification>
