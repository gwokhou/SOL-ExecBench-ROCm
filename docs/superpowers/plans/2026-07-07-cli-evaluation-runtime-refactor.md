# CLI Evaluation Runtime Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract CLI evaluation subprocess orchestration from `src/sol_execbench/cli/main.py` into a focused runtime helper without changing CLI behavior.

**Architecture:** Keep `main.py` responsible for Click options, console/progress presentation, output writing, sidecar emission, and process exit. Add `src/sol_execbench/cli/evaluation_runtime.py` for running the normal/profiled evaluation command and classifying subprocess outcomes into typed results. Reuse existing low-level helpers in `src/sol_execbench/cli/evaluation.py`; do not move no-trace sidecar serialization or rocprofv3 integration internals.

**Tech Stack:** Python 3.12, Click, Rich, Pydantic trace models, `subprocess.CompletedProcess`, pytest, Ruff.

---

## File Structure

- Create `src/sol_execbench/cli/evaluation_runtime.py`
  - Owns runtime-level evaluation execution.
  - Defines immutable result dataclasses for success and no-trace failure cases.
  - Calls `cli_evaluation._run_profiled_evaluation()`, `cli_evaluation._run_evaluation_command()`, `cli_evaluation._timeout_output_text()`, and `filter_benign_rocm_stderr()`.
  - Does not print to console.
  - Does not write files.
  - Does not call `sys.exit()`.

- Modify `src/sol_execbench/cli/main.py`
  - Imports `evaluation_runtime as cli_evaluation_runtime`.
  - Owns eval driver generation by calling `packager.execute()` before invoking runtime.
  - Replaces inline Phase 2 subprocess/profile/timeout/no-stdout/no-traces classification with calls into `run_evaluation_runtime()`.
  - Keeps progress spinner, console messages, no-trace sidecar writing, trace output, sidecar output, and exit behavior.

- Create `tests/sol_execbench/test_cli_evaluation_runtime.py`
  - Unit tests for runtime behavior without GPU/ROCm.
  - Uses fake packagers and monkeypatched runner functions.
  - Tests timeout classification, no-stdout failure classification, no-parseable-traces classification, successful trace parsing, and profiled fallback behavior.

- Modify `tests/sol_execbench/test_cli_module_boundaries.py`
  - Adds a boundary test that runtime helpers live outside `main.py`.

- Keep existing tests unchanged unless a moved responsibility needs import adjustment:
  - `tests/sol_execbench/test_cli_evaluation_timeout.py`
  - `tests/sol_execbench/test_cli_diagnostics.py`
  - `tests/sol_execbench/test_cli_profile_sidecars.py`
  - `tests/sol_execbench/test_cli_static_evidence.py`
  - `tests/sol_execbench/test_cli_agent_feedback_sidecar.py`

---

### Task 1: Add Runtime Module Boundary Tests

**Files:**
- Modify: `tests/sol_execbench/test_cli_module_boundaries.py`

- [ ] **Step 1: Add the failing boundary test**

Append this test to `tests/sol_execbench/test_cli_module_boundaries.py`:

```python
def test_cli_evaluation_runtime_helpers_live_outside_main() -> None:
    from sol_execbench.cli import evaluation_runtime

    assert evaluation_runtime.EvaluationRuntimeSuccess is not None
    assert evaluation_runtime.EvaluationRuntimeNoTraceFailure is not None
    assert evaluation_runtime.run_evaluation_runtime is not None

    for name in (
        "EvaluationRuntimeSuccess",
        "EvaluationRuntimeNoTraceFailure",
        "run_evaluation_runtime",
    ):
        assert not hasattr(cli_main, name)
```

- [ ] **Step 2: Run the boundary test and verify it fails**

Run:

```bash
uv run pytest tests/sol_execbench/test_cli_module_boundaries.py::test_cli_evaluation_runtime_helpers_live_outside_main -q
```

Expected:

```text
FAILED ... ImportError: cannot import name 'evaluation_runtime' from 'sol_execbench.cli'
```

- [ ] **Step 3: Commit the failing boundary test**

Run:

```bash
git add tests/sol_execbench/test_cli_module_boundaries.py
git commit -s -m "#0 - Add CLI evaluation runtime boundary tests"
```

---

### Task 2: Add Runtime Unit Tests Before Implementation

**Files:**
- Create: `tests/sol_execbench/test_cli_evaluation_runtime.py`

- [ ] **Step 1: Create test scaffolding**

Create `tests/sol_execbench/test_cli_evaluation_runtime.py`:

```python
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest

from sol_execbench.cli import evaluation as cli_evaluation
from sol_execbench.cli import evaluation_runtime
from sol_execbench.core import EvaluationStatus
from sol_execbench.core.bench.rocm_profiler import Rocprofv3ProfileResult


class _FakeTrace:
    def __init__(self, status: EvaluationStatus = EvaluationStatus.PASSED) -> None:
        self.evaluation = type("Evaluation", (), {"status": status})()

    def model_dump(self, *, mode: str) -> dict[str, Any]:
        assert mode == "json"
        return {"evaluation": {"status": self.evaluation.status.value}}


class _FakePackager:
    def __init__(self, traces: list[_FakeTrace] | None = None) -> None:
        self.traces = traces or []
        self.converted_stdout: str | None = None

    def execute(self) -> list[str]:
        raise AssertionError("runtime must not call execute")

    def convert_stdout_to_traces(self, stdout: str) -> list[_FakeTrace]:
        self.converted_stdout = stdout
        return self.traces
```

- [ ] **Step 2: Add success test**

Add:

```python
def test_run_evaluation_runtime_returns_success_for_parseable_traces(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    packager = _FakePackager(traces=[_FakeTrace()])

    def _run_command(eval_cmd, *, staging_dir, timeout):  # noqa: ANN001, ARG001
        return subprocess.CompletedProcess(
            args=eval_cmd,
            returncode=0,
            stdout='{"trace": 1}\n',
            stderr="",
        )

    monkeypatch.setattr(cli_evaluation, "_run_evaluation_command", _run_command)

    result = evaluation_runtime.run_evaluation_runtime(
        packager,
        eval_cmd=["python", "candidate.py"],
        staging_dir=tmp_path,
        output_file=None,
        timeout=7,
        profile="none",
    )

    assert isinstance(result, evaluation_runtime.EvaluationRuntimeSuccess)
    assert packager.converted_stdout == '{"trace": 1}\n'
    assert len(result.traces) == 1
    assert result.returncode == 0
    assert result.filtered_stderr == ""
    assert result.profile_result is None
```

- [ ] **Step 3: Add timeout classification test**

Add:

```python
def test_run_evaluation_runtime_classifies_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    packager = _FakePackager()

    def _raise_timeout(eval_cmd, *, staging_dir, timeout):  # noqa: ANN001, ARG001
        raise subprocess.TimeoutExpired(
            cmd=eval_cmd,
            timeout=timeout,
            output=b"partial stdout",
            stderr=b"partial stderr",
        )

    monkeypatch.setattr(cli_evaluation, "_run_evaluation_command", _raise_timeout)

    result = evaluation_runtime.run_evaluation_runtime(
        packager,
        eval_cmd=["python", "candidate.py"],
        staging_dir=tmp_path,
        output_file=None,
        timeout=5,
        profile="none",
    )

    assert isinstance(result, evaluation_runtime.EvaluationRuntimeNoTraceFailure)
    assert result.reason == "evaluation_timeout"
    assert result.returncode == 124
    assert result.stdout == "partial stdout"
    assert result.stderr == "partial stderr"
    assert result.filtered_stderr == "partial stderr"
    assert result.message == "Evaluation timed out after 5s"
```

- [ ] **Step 4: Add no-stdout failure classification test**

Add:

```python
def test_run_evaluation_runtime_classifies_failure_without_stdout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    packager = _FakePackager()

    def _run_command(eval_cmd, *, staging_dir, timeout):  # noqa: ANN001, ARG001
        return subprocess.CompletedProcess(
            args=eval_cmd,
            returncode=2,
            stdout=" \n",
            stderr="real error",
        )

    monkeypatch.setattr(cli_evaluation, "_run_evaluation_command", _run_command)

    result = evaluation_runtime.run_evaluation_runtime(
        packager,
        eval_cmd=["python", "candidate.py"],
        staging_dir=tmp_path,
        output_file=None,
        timeout=5,
        profile="none",
    )

    assert isinstance(result, evaluation_runtime.EvaluationRuntimeNoTraceFailure)
    assert result.reason == "evaluation_failed_no_stdout"
    assert result.returncode == 2
    assert result.stdout == " \n"
    assert result.stderr == "real error"
    assert result.filtered_stderr == "real error"
    assert result.message == "Evaluation failed"
```

- [ ] **Step 5: Add no-parseable-traces classification test**

Add:

```python
def test_run_evaluation_runtime_classifies_no_parseable_traces(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    packager = _FakePackager(traces=[])

    def _run_command(eval_cmd, *, staging_dir, timeout):  # noqa: ANN001, ARG001
        return subprocess.CompletedProcess(
            args=eval_cmd,
            returncode=0,
            stdout="not json traces",
            stderr="warning",
        )

    monkeypatch.setattr(cli_evaluation, "_run_evaluation_command", _run_command)

    result = evaluation_runtime.run_evaluation_runtime(
        packager,
        eval_cmd=["python", "candidate.py"],
        staging_dir=tmp_path,
        output_file=None,
        timeout=5,
        profile="none",
    )

    assert isinstance(result, evaluation_runtime.EvaluationRuntimeNoTraceFailure)
    assert result.reason == "no_parseable_traces"
    assert result.returncode == 0
    assert result.stdout == "not json traces"
    assert result.stderr == "warning"
    assert result.filtered_stderr == "warning"
    assert result.message == "No traces produced"
```

- [ ] **Step 6: Add profiled fallback test**

Add:

```python
def test_run_evaluation_runtime_falls_back_when_profile_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    packager = _FakePackager(traces=[_FakeTrace()])
    profile_result = Rocprofv3ProfileResult(
        command=("rocprofv3",),
        output_directory=tmp_path,
        succeeded=False,
        skipped_reason="rocprofv3 unavailable",
    )

    def _run_profiled(eval_cmd, *, staging_dir, output_file, timeout):  # noqa: ANN001, ARG001
        return None, profile_result

    def _run_command(eval_cmd, *, staging_dir, timeout):  # noqa: ANN001, ARG001
        return subprocess.CompletedProcess(
            args=eval_cmd,
            returncode=0,
            stdout='{"trace": 1}\n',
            stderr="",
        )

    monkeypatch.setattr(cli_evaluation, "_run_profiled_evaluation", _run_profiled)
    monkeypatch.setattr(cli_evaluation, "_run_evaluation_command", _run_command)

    result = evaluation_runtime.run_evaluation_runtime(
        packager,
        eval_cmd=["python", "candidate.py"],
        staging_dir=tmp_path,
        output_file=tmp_path / "trace.jsonl",
        timeout=5,
        profile="rocprofv3",
    )

    assert isinstance(result, evaluation_runtime.EvaluationRuntimeSuccess)
    assert result.profile_result is profile_result
    assert result.profile_fallback_reason == "rocprofv3 unavailable"
```

- [ ] **Step 7: Run runtime tests and verify they fail before implementation**

Run:

```bash
uv run pytest tests/sol_execbench/test_cli_evaluation_runtime.py -q
```

Expected:

```text
FAILED ... ImportError: cannot import name 'evaluation_runtime' from 'sol_execbench.cli'
```

- [ ] **Step 8: Commit failing runtime tests**

Run:

```bash
git add tests/sol_execbench/test_cli_evaluation_runtime.py
git commit -s -m "#0 - Add CLI evaluation runtime tests"
```

---

### Task 3: Implement `evaluation_runtime.py`

**Files:**
- Create: `src/sol_execbench/cli/evaluation_runtime.py`

- [ ] **Step 1: Add module header, imports, and result dataclasses**

Create `src/sol_execbench/cli/evaluation_runtime.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Evaluation runtime orchestration helpers for the SOL-ExecBench CLI."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

from . import evaluation as cli_evaluation
from ..core.bench.rocm_profiler import Rocprofv3ProfileResult
from ..core.bench.stderr import filter_benign_rocm_stderr

PROFILE_NONE = "none"
PROFILE_ROCPROFV3 = "rocprofv3"


class EvaluationPackager(Protocol):
    def convert_stdout_to_traces(self, stdout: str) -> list[Any]: ...


@dataclass(frozen=True)
class EvaluationRuntimeSuccess:
    traces: list[Any]
    stdout: str
    stderr: str
    filtered_stderr: str
    returncode: int
    profile_result: Rocprofv3ProfileResult | None
    profile_fallback_reason: str | None = None


@dataclass(frozen=True)
class EvaluationRuntimeNoTraceFailure:
    reason: Literal[
        "evaluation_timeout",
        "evaluation_failed_no_stdout",
        "no_parseable_traces",
    ]
    message: str
    stdout: str
    stderr: str
    filtered_stderr: str
    returncode: int
    profile_result: Rocprofv3ProfileResult | None
    profile_fallback_reason: str | None = None
```

- [ ] **Step 2: Add profile execution helper**

Append:

```python
def _profile_fallback_reason(profile_result: Rocprofv3ProfileResult | None) -> str | None:
    if profile_result is None:
        return None
    return profile_result.skipped_reason or profile_result.failed_reason


def _run_profiled_or_none(
    eval_cmd: list[str],
    *,
    staging_dir: Path,
    output_file: Path | None,
    timeout: int,
    profile: str,
) -> tuple[subprocess.CompletedProcess[str] | None, Rocprofv3ProfileResult | None]:
    if profile != PROFILE_ROCPROFV3:
        return None, None
    return cli_evaluation._run_profiled_evaluation(
        eval_cmd,
        staging_dir=staging_dir,
        output_file=output_file,
        timeout=timeout,
    )
```

- [ ] **Step 3: Add `run_evaluation_runtime()`**

Append:

```python
def run_evaluation_runtime(
    packager: EvaluationPackager,
    *,
    eval_cmd: list[str],
    staging_dir: Path,
    output_file: Path | None,
    timeout: int,
    profile: str,
) -> EvaluationRuntimeSuccess | EvaluationRuntimeNoTraceFailure:
    """Run evaluation and classify subprocess outcomes without CLI side effects."""

    profiled_proc, profile_result = _run_profiled_or_none(
        eval_cmd,
        staging_dir=staging_dir,
        output_file=output_file,
        timeout=timeout,
        profile=profile,
    )
    fallback_reason = _profile_fallback_reason(profile_result)

    try:
        proc = profiled_proc or cli_evaluation._run_evaluation_command(
            eval_cmd,
            staging_dir=staging_dir,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = cli_evaluation._timeout_output_text(exc.stdout)
        stderr = cli_evaluation._timeout_output_text(exc.stderr)
        return EvaluationRuntimeNoTraceFailure(
            reason="evaluation_timeout",
            message=f"Evaluation timed out after {timeout}s",
            returncode=124,
            stdout=stdout,
            stderr=stderr,
            filtered_stderr=filter_benign_rocm_stderr(stderr),
            profile_result=profile_result,
            profile_fallback_reason=fallback_reason,
        )

    filtered_stderr = filter_benign_rocm_stderr(proc.stderr)
    if proc.returncode != 0 and not proc.stdout.strip():
        return EvaluationRuntimeNoTraceFailure(
            reason="evaluation_failed_no_stdout",
            message="Evaluation failed",
            returncode=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
            filtered_stderr=filtered_stderr,
            profile_result=profile_result,
            profile_fallback_reason=fallback_reason,
        )

    traces = packager.convert_stdout_to_traces(proc.stdout)
    if not traces:
        return EvaluationRuntimeNoTraceFailure(
            reason="no_parseable_traces",
            message="No traces produced",
            returncode=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
            filtered_stderr=filtered_stderr,
            profile_result=profile_result,
            profile_fallback_reason=fallback_reason,
        )

    return EvaluationRuntimeSuccess(
        traces=traces,
        stdout=proc.stdout,
        stderr=proc.stderr,
        filtered_stderr=filtered_stderr,
        returncode=proc.returncode,
        profile_result=profile_result,
        profile_fallback_reason=fallback_reason,
    )
```

- [ ] **Step 4: Run runtime and boundary tests**

Run:

```bash
uv run pytest tests/sol_execbench/test_cli_evaluation_runtime.py tests/sol_execbench/test_cli_module_boundaries.py -q
```

Expected:

```text
... passed ...
```

- [ ] **Step 5: Commit runtime helper**

Run:

```bash
git add src/sol_execbench/cli/evaluation_runtime.py
git commit -s -m "#0 - Extract CLI evaluation runtime helper"
```

---

### Task 4: Wire Runtime Helper Into `main.py`

**Files:**
- Modify: `src/sol_execbench/cli/main.py`
- Modify: `tests/sol_execbench/test_cli_evaluation_timeout.py`

- [ ] **Step 1: Update imports in `main.py`**

Change the import block:

```python
import subprocess
```

Remove it if no longer used after wiring.

Add:

```python
from . import evaluation_runtime as cli_evaluation_runtime
```

Keep:

```python
from . import evaluation as cli_evaluation
```

only if `main.py` still calls `_write_no_trace_diagnostics_sidecar()` directly.

- [ ] **Step 2: Replace Phase 2 inline subprocess logic**

Replace the current Phase 2 block from:

```python
# Phase 2: Evaluate
eval_cmd = packager.execute()

profile_result: Rocprofv3ProfileResult | None = None
profiled_proc: subprocess.CompletedProcess[str] | None = None
if profile == PROFILE_ROCPROFV3:
    ...
```

through the end of the no-trace checks with:

```python
# Phase 2: Evaluate
with Progress(
    SpinnerColumn(),
    TextColumn("[progress.description]{task.description}"),
    console=console,
) as progress:
    task = progress.add_task(
        f"Evaluating {len(workloads)} workload(s)...", total=None
    )
    runtime_result = cli_evaluation_runtime.run_evaluation_runtime(
        packager,
        staging_dir=staging_dir,
        output_file=output_file,
        timeout=timeout,
        profile=profile,
    )
    progress.update(task, completed=True)

profile_result = runtime_result.profile_result
if runtime_result.profile_fallback_reason is not None:
    console.print(
        "[yellow]rocprofv3 profiling unavailable or failed; "
        "running normal evaluation. Reason: "
        f"{runtime_result.profile_fallback_reason}[/yellow]"
    )

if runtime_result.filtered_stderr and verbose:
    console.print(f"[dim]{runtime_result.filtered_stderr}[/dim]")

if isinstance(runtime_result, cli_evaluation_runtime.EvaluationRuntimeNoTraceFailure):
    console.print(f"[red]{runtime_result.message}[/red]")
    diagnostic_path = cli_evaluation._write_no_trace_diagnostics_sidecar(
        output_file=output_file,
        staging_dir=staging_dir,
        keep_staging=keep_staging,
        reason=runtime_result.reason,
        returncode=runtime_result.returncode,
        stdout=runtime_result.stdout,
        stderr=runtime_result.stderr,
    )
    if diagnostic_path is not None:
        console.print(f"[yellow]Saved no-trace diagnostics to {diagnostic_path}[/yellow]")
    if runtime_result.filtered_stderr:
        console.print(runtime_result.filtered_stderr)
    packager.close()
    sys.exit(1)

traces = runtime_result.traces
```

- [ ] **Step 3: Preserve timeout CLI monkeypatch behavior**

Update `tests/sol_execbench/test_cli_evaluation_timeout.py` so it monkeypatches the low-level helper that runtime calls:

```python
monkeypatch.setattr(cli_evaluation, "_run_evaluation_command", _raise_timeout)
```

Keep this line unchanged if the runtime helper imports `cli_evaluation` as a module, because monkeypatching the module attribute still affects runtime calls.

- [ ] **Step 4: Run CLI runtime regression tests**

Run:

```bash
uv run pytest tests/sol_execbench/test_cli_evaluation_runtime.py tests/sol_execbench/test_cli_evaluation_timeout.py tests/sol_execbench/test_cli_module_boundaries.py -q
```

Expected:

```text
... passed ...
```

- [ ] **Step 5: Commit main wiring**

Run:

```bash
git add src/sol_execbench/cli/main.py tests/sol_execbench/test_cli_evaluation_timeout.py
git commit -s -m "#0 - Wire CLI evaluation runtime helper"
```

---

### Task 5: Focused Regression Verification

**Files:**
- No planned source edits.
- Modify only directly affected files if Ruff reports an issue.

- [ ] **Step 1: Run runtime and boundary tests**

Run:

```bash
uv run pytest tests/sol_execbench/test_cli_evaluation_runtime.py tests/sol_execbench/test_cli_evaluation_timeout.py tests/sol_execbench/test_cli_module_boundaries.py -q
```

Expected:

```text
... passed ...
```

- [ ] **Step 2: Run CLI evaluation-adjacent regression tests**

Run:

```bash
uv run pytest tests/sol_execbench/test_cli_diagnostics.py tests/sol_execbench/test_cli_environment.py tests/sol_execbench/test_cli_reporting.py tests/sol_execbench/test_cli_problem_io.py tests/sol_execbench/test_cli_compilation.py -q
```

Expected:

```text
... passed ...
```

- [ ] **Step 3: Run sidecar regression tests**

Run:

```bash
uv run pytest tests/sol_execbench/test_cli_profile_sidecars.py tests/sol_execbench/test_cli_static_evidence.py tests/sol_execbench/test_cli_agent_feedback_sidecar.py -q
```

Expected:

```text
... passed ...
```

- [ ] **Step 4: Run targeted Ruff**

Run:

```bash
uv run --with ruff ruff check src/sol_execbench/cli/main.py src/sol_execbench/cli/evaluation.py src/sol_execbench/cli/evaluation_runtime.py tests/sol_execbench/test_cli_evaluation_runtime.py tests/sol_execbench/test_cli_evaluation_timeout.py tests/sol_execbench/test_cli_module_boundaries.py
```

Expected:

```text
All checks passed!
```

- [ ] **Step 5: Commit verification cleanup only if files changed**

If Ruff required edits, run:

```bash
git add src/sol_execbench/cli/main.py src/sol_execbench/cli/evaluation_runtime.py tests/sol_execbench/test_cli_evaluation_runtime.py tests/sol_execbench/test_cli_evaluation_timeout.py tests/sol_execbench/test_cli_module_boundaries.py
git commit -s -m "#0 - Verify CLI evaluation runtime refactor"
```

If no files changed, do not create an empty commit.

---

### Task 6: Plan Artifact Commit and Final Review

**Files:**
- Add: `docs/superpowers/plans/2026-07-07-cli-evaluation-runtime-refactor.md`

- [ ] **Step 1: Commit this plan document**

Run:

```bash
git add docs/superpowers/plans/2026-07-07-cli-evaluation-runtime-refactor.md
git commit -s -m "#0 - Add CLI evaluation runtime refactor plan"
```

- [ ] **Step 2: Run final review**

Request a read-only review over:

```text
src/sol_execbench/cli/main.py
src/sol_execbench/cli/evaluation.py
src/sol_execbench/cli/evaluation_runtime.py
tests/sol_execbench/test_cli_evaluation_runtime.py
tests/sol_execbench/test_cli_evaluation_timeout.py
tests/sol_execbench/test_cli_module_boundaries.py
docs/superpowers/plans/2026-07-07-cli-evaluation-runtime-refactor.md
```

The review should focus on:

```text
Behavior regressions in timeout/no-stdout/no-trace handling.
Profile fallback message equivalence.
Verbose stderr output equivalence.
Packager close and sys.exit behavior remaining in main.
No hidden file writes or sys.exit calls inside evaluation_runtime.py.
Test brittleness or stale imports.
```

- [ ] **Step 3: Run final completion verification**

Run:

```bash
uv run pytest tests/sol_execbench/test_cli_evaluation_runtime.py tests/sol_execbench/test_cli_evaluation_timeout.py tests/sol_execbench/test_cli_module_boundaries.py -q
uv run pytest tests/sol_execbench/test_cli_diagnostics.py tests/sol_execbench/test_cli_environment.py tests/sol_execbench/test_cli_reporting.py tests/sol_execbench/test_cli_problem_io.py tests/sol_execbench/test_cli_compilation.py -q
uv run pytest tests/sol_execbench/test_cli_profile_sidecars.py tests/sol_execbench/test_cli_static_evidence.py tests/sol_execbench/test_cli_agent_feedback_sidecar.py -q
uv run --with ruff ruff check src/sol_execbench/cli/main.py src/sol_execbench/cli/evaluation.py src/sol_execbench/cli/evaluation_runtime.py tests/sol_execbench/test_cli_evaluation_runtime.py tests/sol_execbench/test_cli_evaluation_timeout.py tests/sol_execbench/test_cli_module_boundaries.py
git status --short
git branch --show-current
```

Expected:

```text
All pytest commands pass.
Ruff prints: All checks passed!
git status --short prints nothing.
git branch --show-current prints: main
```

---

## Self-Review

- Spec coverage: The plan covers boundary tests, runtime unit tests, implementation, main wiring, regression verification, plan artifact commit, and final review.
- Placeholder scan: No `TBD`, `TODO`, "implement later", or unspecified edge-case steps remain.
- Type consistency: `EvaluationRuntimeSuccess`, `EvaluationRuntimeNoTraceFailure`, and `run_evaluation_runtime()` names are consistent across tests, implementation, boundary checks, and main wiring.
- Scope check: The plan does not move sidecar writing, output writing, Click command registration, compile logic, problem input loading, or low-level profile/no-trace serialization internals.
