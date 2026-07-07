# Dataset Test Boundaries Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split `test_dataset_runner.py` so tests live beside the production module boundary they exercise.

**Architecture:** `test_dataset_solutions.py` owns solution helper tests, `test_cli_execution.py` owns CLI execution/logging tests, and `test_dataset_runner.py` keeps runner summary/reporting tests. No production code changes.

**Tech Stack:** Python 3.12, pytest, Ruff.

---

## File Structure

- Create `tests/sol_execbench/test_dataset_solutions.py`: solution sanitizer and Solution dict construction tests.
- Create `tests/sol_execbench/test_cli_execution.py`: CLI subprocess/logging/failure-note tests.
- Modify `tests/sol_execbench/test_dataset_runner.py`: keep only runner summary/reporting tests.

## Task 1: Move Dataset Solution Tests

**Files:**
- Create: `tests/sol_execbench/test_dataset_solutions.py`
- Modify: `tests/sol_execbench/test_dataset_runner.py`

- [ ] **Step 1: Create `test_dataset_solutions.py`**

Move these from `tests/sol_execbench/test_dataset_runner.py` into the new file:

```python
def _definition(reference: str = "def run(x):\n    return x\n") -> dict: ...
def test_sanitize_python_source_only_rewrites_stream_identifiers(): ...
def test_build_reference_solution_uses_token_aware_stream_sanitizer(): ...
def test_build_custom_solution_preserves_metadata_and_detects_dps(tmp_path): ...
```

The new file should have:

```python
from __future__ import annotations

from sol_execbench.core.dataset import solutions
```

Keep test bodies and assertions behavior-equivalent.

- [ ] **Step 2: Remove moved solution tests from `test_dataset_runner.py`**

Delete the moved `_definition` helper and three solution tests from `test_dataset_runner.py`.

Remove the now-unused import:

```python
from sol_execbench.core.dataset import solutions
```

Keep `runner` import for remaining summary tests.

- [ ] **Step 3: Run focused tests**

Run:

```bash
uv run pytest tests/sol_execbench/test_dataset_solutions.py tests/sol_execbench/test_dataset_runner.py -v
```

Expected:

```text
... passed
```

- [ ] **Step 4: Commit**

```bash
git add tests/sol_execbench/test_dataset_solutions.py tests/sol_execbench/test_dataset_runner.py
git commit -s -m "#0 - Move dataset solution tests"
```

## Task 2: Move CLI Execution Tests

**Files:**
- Create: `tests/sol_execbench/test_cli_execution.py`
- Modify: `tests/sol_execbench/test_dataset_runner.py`

- [ ] **Step 1: Create `test_cli_execution.py`**

Move these from `tests/sol_execbench/test_dataset_runner.py` into the new file:

```python
def test_run_cli_parses_jsonl_and_ignores_non_json_stdout(tmp_path, monkeypatch): ...
def test_run_cli_passes_flashinfer_safetensors_env(tmp_path, monkeypatch): ...
def test_run_cli_writes_log_for_nonzero_exit(tmp_path, monkeypatch): ...
def test_run_cli_log_filters_benign_amdgpu_ids_noise(tmp_path, monkeypatch): ...
def test_run_cli_writes_log_for_timeout(tmp_path, monkeypatch): ...
```

The new file should have:

```python
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from sol_execbench.core.dataset import cli_execution
```

Keep test bodies and assertions behavior-equivalent.

- [ ] **Step 2: Remove moved CLI tests from `test_dataset_runner.py`**

Delete the moved CLI tests from `test_dataset_runner.py`.

Remove imports that are no longer needed there:

```python
import subprocess
from pathlib import Path
from sol_execbench.core.dataset import cli_execution
```

Keep `json` if `test_dataset_runner.py` still uses it for summary report assertions.

- [ ] **Step 3: Run focused tests**

Run:

```bash
uv run pytest tests/sol_execbench/test_cli_execution.py tests/sol_execbench/test_dataset_runner.py -v
```

Expected:

```text
... passed
```

- [ ] **Step 4: Commit**

```bash
git add tests/sol_execbench/test_cli_execution.py tests/sol_execbench/test_dataset_runner.py
git commit -s -m "#0 - Move CLI execution tests"
```

## Task 3: Regression Verification

**Files:**
- Verify: `tests/sol_execbench/test_dataset_runner.py`
- Verify: `tests/sol_execbench/test_dataset_solutions.py`
- Verify: `tests/sol_execbench/test_cli_execution.py`
- Verify: `tests/sol_execbench/test_run_dataset_execution_closure.py`
- Verify: `tests/sol_execbench/test_run_dataset_amd_score.py`

- [ ] **Step 1: Run split test files**

```bash
uv run pytest tests/sol_execbench/test_dataset_runner.py tests/sol_execbench/test_dataset_solutions.py tests/sol_execbench/test_cli_execution.py
```

Expected:

```text
... passed
```

- [ ] **Step 2: Run dependent dataset regressions**

```bash
uv run pytest tests/sol_execbench/test_run_dataset_execution_closure.py tests/sol_execbench/test_run_dataset_amd_score.py
```

Expected:

```text
... passed
```

- [ ] **Step 3: Run Ruff on changed test files**

```bash
uv run --with ruff ruff check tests/sol_execbench/test_dataset_runner.py tests/sol_execbench/test_dataset_solutions.py tests/sol_execbench/test_cli_execution.py
```

Expected:

```text
All checks passed!
```

- [ ] **Step 4: Commit verification fixes if needed**

If Task 3 required no file changes, do not create a commit. If verification required fixes, commit them:

```bash
git add <fixed-files>
git commit -s -m "#0 - Fix dataset test boundary regressions"
```

## Self-Review

- Spec coverage: Tasks move solution tests, CLI execution tests, and run the requested regression commands.
- Placeholder scan: No placeholder steps remain; file paths, test names, and commands are explicit.
- Type consistency: All moved tests keep existing pytest fixtures and assertions.
