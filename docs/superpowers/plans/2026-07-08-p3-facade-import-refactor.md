# P3 Facade Import Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce P3 facade coupling by removing internal production imports from `sol_execbench.core` and `sol_execbench.core.data` root re-export modules.

**Architecture:** Keep public facade modules intact for external compatibility. Update internal source files to import directly from concrete schema/config modules, then enforce this with AST boundary tests.

**Tech Stack:** Python 3.12+, pytest, Ruff, AST-based module boundary tests.

---

### Task 1: Add P3 Facade Guardrails

**Files:**
- Modify: `tests/sol_execbench/cli/test_module_boundaries.py`

- [ ] Add a test that scans non-`__init__.py` source modules and rejects `from sol_execbench.core import ...` and `from sol_execbench.core.data import ...`.
- [ ] Run the boundary test and confirm it fails against the current internal facade imports.

### Task 2: Replace `sol_execbench.core` Root Imports

**Files:**
- Modify source files under `src/sol_execbench/core/bench/`, `src/sol_execbench/driver/`, and `src/sol_execbench/cli/`.

- [ ] Replace `BenchmarkConfig` imports with `sol_execbench.core.bench.config`.
- [ ] Replace `Definition` imports with `sol_execbench.core.data.definition`.
- [ ] Replace `Solution`/`SupportedHardware` imports with `sol_execbench.core.data.solution`.
- [ ] Replace `Workload` imports with `sol_execbench.core.data.workload`.
- [ ] Replace `Trace`, `EvaluationStatus`, `Correctness`, `Performance` imports with `sol_execbench.core.data.trace`.

### Task 3: Replace `sol_execbench.core.data` Root Imports

**Files:**
- Modify source files under `src/sol_execbench/core/bench/` and `src/sol_execbench/core/utils.py`.

- [ ] Replace data facade imports with concrete `definition`, `trace`, and `workload` module imports.
- [ ] Preserve public re-export behavior of `core/__init__.py` and `core/data/__init__.py`.

### Task 4: Verify

**Files:**
- No code changes expected.

- [ ] Run `uv run pytest tests/sol_execbench/cli/test_module_boundaries.py -q`.
- [ ] Run focused CLI/bench/driver tests affected by import rewrites.
- [ ] Run Ruff on touched files.
- [ ] Run `git diff --check`.
