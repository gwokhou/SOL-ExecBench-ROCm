# P2 Coupling Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce P2 coupling hotspots while preserving public imports and benchmark behavior.

**Architecture:** Keep existing public modules as facades. Move builder, governance, citation, heuristic tensor, and FX helper code into focused sibling modules with explicit imports.

**Tech Stack:** Python 3.12+, pytest, Ruff, AST-based module boundary tests.

---

### Task 1: Add P2 Coupling Guardrails

**Files:**
- Modify: `tests/sol_execbench/cli/test_module_boundaries.py`

- [ ] Add tests bounding P2 files by direct internal import fan-out and line count.
- [ ] Run `uv run pytest tests/sol_execbench/cli/test_module_boundaries.py -q`.
- [ ] Confirm the new tests fail against the current P2 hotspots.

### Task 2: Split Agent Feedback Facade

**Files:**
- Modify: `src/sol_execbench/core/bench/agent_feedback.py`
- Create: `src/sol_execbench/core/bench/agent_feedback_builder.py`
- Create: `src/sol_execbench/core/bench/agent_feedback_governance.py`
- Create: `src/sol_execbench/core/bench/agent_feedback_artifacts.py`

- [ ] Move sidecar construction and aggregate helper functions into `agent_feedback_builder.py`.
- [ ] Move freshness and governance functions into `agent_feedback_governance.py`.
- [ ] Move artifact citation helper into `agent_feedback_artifacts.py`.
- [ ] Keep `agent_feedback.py` re-export compatibility for existing tests and callers.

### Task 3: Split Profile Summary Facade

**Files:**
- Modify: `src/sol_execbench/core/bench/profile_summary.py`
- Create: `src/sol_execbench/core/bench/profile_summary_sidecar_models.py`
- Create: `src/sol_execbench/core/bench/profile_summary_builder.py`
- Create: `src/sol_execbench/core/bench/profile_summary_governance.py`
- Create: `src/sol_execbench/core/bench/profile_summary_citations.py`

- [ ] Move profile summary sidecar enums/models into `profile_summary_sidecar_models.py`.
- [ ] Move sidecar construction/content/limitations helpers into `profile_summary_builder.py`.
- [ ] Move freshness and governance functions into `profile_summary_governance.py`.
- [ ] Move artifact citation helper into `profile_summary_citations.py`.
- [ ] Keep `profile_summary.py` as a public compatibility facade.

### Task 4: Split Input Heuristics

**Files:**
- Modify: `src/sol_execbench/core/bench/input_generation.py`
- Create: `src/sol_execbench/core/bench/input_heuristics.py`

- [ ] Move heuristic tensor detection and generation functions into `input_heuristics.py`.
- [ ] Re-export `_generate_heuristic_tensor` from `input_generation.py` for existing tests.
- [ ] Keep `gen_inputs` behavior unchanged.

### Task 5: Split FX Helper Logic

**Files:**
- Modify: `src/sol_execbench/core/scoring/amd_bound_graph_fx.py`
- Create: `src/sol_execbench/core/scoring/amd_bound_graph_fx_helpers.py`

- [ ] Move `_torch_dtype`, node naming, input/output flattening, source expression, attributes, and first-input fallback helpers into `amd_bound_graph_fx_helpers.py`.
- [ ] Keep `_try_fx_bound_graph` as the public internal entry point used by the graph builder.

### Task 6: Verify

**Files:**
- No code changes expected.

- [ ] Run `uv run pytest tests/sol_execbench/cli/test_module_boundaries.py -q`.
- [ ] Run focused tests:
  `uv run pytest tests/sol_execbench/core/bench/test_agent_feedback.py tests/sol_execbench/core/bench/test_agent_feedback_models.py tests/sol_execbench/core/bench/test_profile_summary.py tests/sol_execbench/core/bench/test_profile_summary_models.py tests/sol_execbench/core/bench/test_io.py tests/sol_execbench/core/scoring/test_amd_bound_graph.py -q`.
- [ ] Run Ruff on touched files.
- [ ] Run `git diff --check`.
