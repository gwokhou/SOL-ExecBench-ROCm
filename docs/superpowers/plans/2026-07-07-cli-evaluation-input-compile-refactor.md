# CLI Evaluation Input and Compile Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce `_evaluate_cli` complexity by extracting problem input/model loading and HIP/C++ compilation into focused CLI modules without changing evaluator behavior.

**Architecture:** Move pure path/model loading helpers into `cli/problem_io.py`, then move compile-phase subprocess execution into `cli/compilation.py` with a structured result object. Keep `_evaluate_cli` as the authority for user-facing flow control, sidecar ordering, `packager.close()`, and `sys.exit(...)`.

**Tech Stack:** Python 3.12, Click exceptions, dataclasses, subprocess, Rich progress output, pytest, existing SOL ExecBench core models and `ProblemPackager`.

---

## File Structure

**Create:**
- `src/sol_execbench/cli/problem_io.py`: problem directory resolution and JSON model loading helpers.
- `src/sol_execbench/cli/compilation.py`: HIP/C++ compile phase execution helper and structured result dataclass.
- `tests/sol_execbench/test_cli_problem_io.py`: focused tests for problem path resolution and model loading.
- `tests/sol_execbench/test_cli_compilation.py`: focused tests for compile command execution, ROCm stderr filtering, and non-C++/failure behavior.

**Modify:**
- `src/sol_execbench/cli/main.py`: import new modules, remove moved helpers, and call compile helper while preserving exit behavior.
- `tests/sol_execbench/test_cli_module_boundaries.py`: assert input/compile helpers live outside `main.py`.

---

## Behavioral Constraints

- Existing CLI syntax remains unchanged.
- `_evaluate_cli` remains responsible for:
  - Click option definitions.
  - user-facing problem/solution/workload/config banner printing.
  - staging directory creation.
  - static evidence collection timing.
  - evaluation subprocess, no-trace diagnostics, sidecar writing, JSON output, human reporting, and exit code.
  - calling `packager.close()` and `sys.exit(...)`.
- Compile failure behavior remains unchanged:
  - print `[red]Compilation failed[/red]`.
  - print filtered stderr if present.
  - print stdout if present.
  - close packager.
  - exit code `1`.
- Compile success behavior remains unchanged:
  - print `[green]Compilation succeeded[/green]`.
  - in verbose mode, print filtered stderr as dim output if present.
- Static evidence collection still happens in `_evaluate_cli` after compile success for C++ and immediately for unsupported non-C++ auto mode.
- No GSD commands or `.planning/` workflow updates are part of this plan.

---

### Task 1: Add input and compilation module boundary tests

**Files:**
- Modify: `tests/sol_execbench/test_cli_module_boundaries.py`

- [ ] **Step 1: Add boundary tests**

Append these tests to `tests/sol_execbench/test_cli_module_boundaries.py`:

```python
def test_cli_problem_io_helpers_live_outside_main() -> None:
    from sol_execbench.cli import problem_io

    assert problem_io.ResolvedProblemInputs is not None
    assert problem_io._load_definition is not None
    assert problem_io._load_workloads is not None
    assert problem_io._load_solution is not None
    assert problem_io._load_config is not None
    assert problem_io._resolve_problem_dir is not None
    assert problem_io.resolve_problem_inputs is not None

    for name in (
        "ResolvedProblemInputs",
        "_load_definition",
        "_load_workloads",
        "_load_solution",
        "_load_config",
        "_resolve_problem_dir",
        "resolve_problem_inputs",
    ):
        assert not hasattr(cli_main, name)


def test_cli_compilation_helpers_live_outside_main() -> None:
    from sol_execbench.cli import compilation

    assert compilation.CompilePhaseResult is not None
    assert compilation.run_compile_phase is not None

    for name in (
        "CompilePhaseResult",
        "run_compile_phase",
    ):
        assert not hasattr(cli_main, name)
```

- [ ] **Step 2: Run boundary tests to verify they fail**

Run:

```bash
uv run pytest tests/sol_execbench/test_cli_module_boundaries.py -q
```

Expected: FAIL because `sol_execbench.cli.problem_io` and `sol_execbench.cli.compilation` do not exist yet.

- [ ] **Step 3: Commit**

```bash
git add tests/sol_execbench/test_cli_module_boundaries.py
git commit -s -m "#0 - Add CLI evaluation boundary tests"
```

---

### Task 2: Extract problem input and model loading helpers

**Files:**
- Create: `src/sol_execbench/cli/problem_io.py`
- Modify: `src/sol_execbench/cli/main.py`
- Create: `tests/sol_execbench/test_cli_problem_io.py`
- Modify: `tests/sol_execbench/test_cli_module_boundaries.py`

- [ ] **Step 1: Create focused problem IO tests**

Create `tests/sol_execbench/test_cli_problem_io.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

import click

from sol_execbench.cli import problem_io
from sol_execbench.core import BenchmarkConfig


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload))


def _definition_payload() -> dict:
    return {
        "name": "toy",
        "description": "toy problem",
        "target": "torch.add",
        "inputs": [
            {
                "name": "x",
                "kind": "tensor",
                "dtype": "float32",
                "shape": ["n"],
            }
        ],
        "outputs": [
            {
                "name": "y",
                "kind": "tensor",
                "dtype": "float32",
                "shape": ["n"],
            }
        ],
    }


def _workload_payload(uuid: str = "w0") -> dict:
    return {
        "uuid": uuid,
        "axes": {"n": 1},
        "inputs": {"n": {"kind": "scalar", "value": 1}},
    }


def _solution_payload() -> dict:
    return {
        "name": "candidate",
        "definition": "toy",
        "author": "agent",
        "spec": {
            "languages": ["pytorch"],
            "target_hardware": ["local"],
            "entry_point": "solution.py::run",
        },
        "sources": [{"path": "solution.py"}],
    }


def test_load_solution_resolves_source_content_relative_to_solution_json(
    tmp_path: Path,
) -> None:
    solution_path = tmp_path / "solution.json"
    (tmp_path / "solution.py").write_text("def run(x):\n    return x\n")
    _write_json(solution_path, _solution_payload())

    solution = problem_io._load_solution(solution_path)

    assert solution.name == "candidate"
    assert solution.sources[0].content == "def run(x):\n    return x\n"


def test_load_workloads_skips_blank_lines(tmp_path: Path) -> None:
    workload_path = tmp_path / "workload.jsonl"
    workload_path.write_text(
        json.dumps(_workload_payload("w0")) + "\n\n" + json.dumps(_workload_payload("w1"))
    )

    workloads = problem_io._load_workloads(workload_path)

    assert [workload.uuid for workload in workloads] == ["w0", "w1"]


def test_load_config_defaults_when_missing() -> None:
    config = problem_io._load_config(None)

    assert isinstance(config, BenchmarkConfig)


def test_load_config_reads_json(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    _write_json(config_path, {"warmup_iters": 3, "benchmark_iters": 7})

    config = problem_io._load_config(config_path)

    assert config.warmup_iters == 3
    assert config.benchmark_iters == 7


def test_resolve_problem_dir_finds_optional_config_and_solution(tmp_path: Path) -> None:
    problem_dir = tmp_path / "problem"
    problem_dir.mkdir()
    for name in ("definition.json", "workload.jsonl", "config.json", "solution.json"):
        (problem_dir / name).write_text("{}")

    definition, workload, config, solution = problem_io._resolve_problem_dir(
        problem_dir
    )

    assert definition == problem_dir / "definition.json"
    assert workload == problem_dir / "workload.jsonl"
    assert config == problem_dir / "config.json"
    assert solution == problem_dir / "solution.json"


def test_resolve_problem_dir_rejects_missing_definition(tmp_path: Path) -> None:
    problem_dir = tmp_path / "problem"
    problem_dir.mkdir()
    (problem_dir / "workload.jsonl").write_text("")

    try:
        problem_io._resolve_problem_dir(problem_dir)
    except click.ClickException as exc:
        assert "definition.json not found" in str(exc)
    else:
        raise AssertionError("expected ClickException")


def test_resolve_problem_inputs_uses_problem_dir_defaults(tmp_path: Path) -> None:
    problem_dir = tmp_path / "problem"
    problem_dir.mkdir()
    definition = problem_dir / "definition.json"
    workload = problem_dir / "workload.jsonl"
    config = problem_dir / "config.json"
    solution = problem_dir / "solution.json"
    definition.write_text("{}")
    workload.write_text("")
    config.write_text("{}")
    solution.write_text("{}")

    resolved = problem_io.resolve_problem_inputs(
        problem_dir=problem_dir,
        definition_file=None,
        workload_file=None,
        solution_file=None,
        config_file=None,
    )

    assert resolved.definition_file == definition
    assert resolved.workload_file == workload
    assert resolved.solution_file == solution
    assert resolved.config_file == config


def test_resolve_problem_inputs_rejects_missing_solution() -> None:
    try:
        problem_io.resolve_problem_inputs(
            problem_dir=None,
            definition_file=Path("definition.json"),
            workload_file=Path("workload.jsonl"),
            solution_file=None,
            config_file=None,
        )
    except click.ClickException as exc:
        assert "Provide PROBLEM_DIR with solution.json or --solution" in str(exc)
    else:
        raise AssertionError("expected ClickException")
```

- [ ] **Step 2: Run problem IO tests to verify they fail**

Run:

```bash
uv run pytest tests/sol_execbench/test_cli_problem_io.py -q
```

Expected: FAIL during import because `sol_execbench.cli.problem_io` does not exist yet.

- [ ] **Step 3: Create `src/sol_execbench/cli/problem_io.py`**

Create `src/sol_execbench/cli/problem_io.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Problem input resolution and model loading helpers for the CLI."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import click

from ..core import BenchmarkConfig, Definition, Solution, Workload


@dataclass(frozen=True)
class ResolvedProblemInputs:
    definition_file: Path
    workload_file: Path
    solution_file: Path
    config_file: Path | None


def _load_definition(path: Path) -> Definition:
    return Definition(**json.loads(path.read_text()))


def _load_workloads(path: Path) -> list[Workload]:
    workloads = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            workloads.append(Workload(**json.loads(line)))
    return workloads


def _load_solution(path: Path) -> Solution:
    sol_dict = json.loads(path.read_text())
    sol_dir = path.parent
    for src in sol_dict.get("sources", []):
        if not src.get("content"):
            src_path = sol_dir / src["path"]
            if src_path.exists():
                src["content"] = src_path.read_text()
    return Solution(**sol_dict)


def _load_config(path: Path | None) -> BenchmarkConfig:
    if path is None:
        return BenchmarkConfig()
    return BenchmarkConfig(**json.loads(path.read_text()))


def _resolve_problem_dir(
    problem_dir: Path,
) -> tuple[Path, Path, Path | None, Path | None]:
    """Return (definition.json, workload.jsonl, config.json?, solution.json?) inside a problem directory."""
    def_path = problem_dir / "definition.json"
    wkl_path = problem_dir / "workload.jsonl"
    cfg_path = problem_dir / "config.json"
    sol_path = problem_dir / "solution.json"
    if not def_path.exists():
        raise click.ClickException(f"definition.json not found in {problem_dir}")
    if not wkl_path.exists():
        raise click.ClickException(f"workload.jsonl not found in {problem_dir}")
    return (
        def_path,
        wkl_path,
        cfg_path if cfg_path.exists() else None,
        sol_path if sol_path.exists() else None,
    )


def resolve_problem_inputs(
    *,
    problem_dir: Path | None,
    definition_file: Path | None,
    workload_file: Path | None,
    solution_file: Path | None,
    config_file: Path | None,
) -> ResolvedProblemInputs:
    if problem_dir:
        def_path, wkl_path, cfg_path, sol_path = _resolve_problem_dir(problem_dir)
        definition_file = definition_file or def_path
        workload_file = workload_file or wkl_path
        config_file = config_file or cfg_path
        solution_file = solution_file or sol_path

    if not definition_file:
        raise click.ClickException("Provide PROBLEM_DIR or --definition")
    if not workload_file:
        raise click.ClickException("Provide PROBLEM_DIR or --workload")
    if not solution_file:
        raise click.ClickException(
            "Provide PROBLEM_DIR with solution.json or --solution"
        )

    return ResolvedProblemInputs(
        definition_file=definition_file,
        workload_file=workload_file,
        solution_file=solution_file,
        config_file=config_file,
    )
```

- [ ] **Step 4: Update `main.py` to use problem_io**

In `src/sol_execbench/cli/main.py`, add:

```python
from . import problem_io as cli_problem_io
```

Remove these moved helper definitions from `main.py`:

```python
_load_definition
_load_workloads
_load_solution
_load_config
_resolve_problem_dir
```

Replace the current problem-dir resolution block in `_evaluate_cli`:

```python
# Resolve definition + workloads
if problem_dir:
    def_path, wkl_path, cfg_path, sol_path = _resolve_problem_dir(problem_dir)
    definition_file = definition_file or def_path
    workload_file = workload_file or wkl_path
    config_file = config_file or cfg_path
    solution_file = solution_file or sol_path

if not definition_file:
    raise click.ClickException("Provide PROBLEM_DIR or --definition")
if not workload_file:
    raise click.ClickException("Provide PROBLEM_DIR or --workload")
if not solution_file:
    raise click.ClickException(
        "Provide PROBLEM_DIR with solution.json or --solution"
    )
```

with:

```python
resolved_inputs = cli_problem_io.resolve_problem_inputs(
    problem_dir=problem_dir,
    definition_file=definition_file,
    workload_file=workload_file,
    solution_file=solution_file,
    config_file=config_file,
)
definition_file = resolved_inputs.definition_file
workload_file = resolved_inputs.workload_file
solution_file = resolved_inputs.solution_file
config_file = resolved_inputs.config_file
```

Replace model loading calls:

```python
definition = _load_definition(definition_file)
workloads = _load_workloads(workload_file)
solution = _load_solution(solution_file)
config = _load_config(config_file)
```

with:

```python
definition = cli_problem_io._load_definition(definition_file)
workloads = cli_problem_io._load_workloads(workload_file)
solution = cli_problem_io._load_solution(solution_file)
config = cli_problem_io._load_config(config_file)
```

Remove imports no longer needed by `main.py` solely for moved helpers:

```python
Optional
Definition
Workload
```

Keep `Optional` only if still used elsewhere in `main.py`.

- [ ] **Step 5: Run problem IO and boundary tests**

Run:

```bash
uv run pytest tests/sol_execbench/test_cli_problem_io.py tests/sol_execbench/test_cli_module_boundaries.py -q
```

Expected: Problem IO tests pass. Boundary tests may still fail only because `compilation` is not extracted until Task 3. If so, run this accepted Task 2 verification instead:

```bash
uv run pytest tests/sol_execbench/test_cli_problem_io.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/sol_execbench/cli/problem_io.py src/sol_execbench/cli/main.py tests/sol_execbench/test_cli_problem_io.py tests/sol_execbench/test_cli_module_boundaries.py
git commit -s -m "#0 - Extract CLI problem input helpers"
```

---

### Task 3: Extract compile phase helper

**Files:**
- Create: `src/sol_execbench/cli/compilation.py`
- Modify: `src/sol_execbench/cli/main.py`
- Create: `tests/sol_execbench/test_cli_compilation.py`
- Modify: `tests/sol_execbench/test_cli_module_boundaries.py`

- [ ] **Step 1: Create focused compilation tests**

Create `tests/sol_execbench/test_cli_compilation.py`:

```python
from __future__ import annotations

import subprocess
from pathlib import Path

from sol_execbench.cli import compilation as cli_compilation


class _FakeProgress:
    def __init__(self) -> None:
        self.added: list[str] = []
        self.completed: list[object] = []

    def add_task(self, description: str, total=None):
        self.added.append(description)
        return "task-1"

    def update(self, task, completed: bool) -> None:
        if completed:
            self.completed.append(task)


class _FakePackager:
    def __init__(self, *, is_cpp: bool = True) -> None:
        self._is_cpp = is_cpp
        self.compile_called = False

    def compile(self):
        self.compile_called = True
        return ["hipcc", "kernel.cpp"], Path("benchmark_kernel.so")


def test_run_compile_phase_skips_non_cpp_solution(tmp_path: Path) -> None:
    packager = _FakePackager(is_cpp=False)
    progress = _FakeProgress()

    result = cli_compilation.run_compile_phase(
        packager=packager,
        staging_dir=tmp_path,
        compile_timeout=30,
        progress=progress,
    )

    assert result.attempted is False
    assert result.succeeded is True
    assert result.artifact_path is None
    assert packager.compile_called is False
    assert progress.added == []


def test_run_compile_phase_executes_compile_command(tmp_path: Path) -> None:
    packager = _FakePackager()
    progress = _FakeProgress()
    captured: dict[str, object] = {}

    def fake_runner(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return subprocess.CompletedProcess(
            args=args[0],
            returncode=0,
            stdout="compiled\n",
            stderr="",
        )

    result = cli_compilation.run_compile_phase(
        packager=packager,
        staging_dir=tmp_path,
        compile_timeout=30,
        progress=progress,
        runner=fake_runner,
        env_builder=lambda env: {**env, "FLASHINFER_TRACE_DIR": "/repo"},
    )

    assert result.attempted is True
    assert result.succeeded is True
    assert result.artifact_path == Path("benchmark_kernel.so")
    assert result.returncode == 0
    assert result.stdout == "compiled\n"
    assert result.filtered_stderr == ""
    assert captured["args"][0] == ["hipcc", "kernel.cpp"]
    assert captured["kwargs"]["cwd"] == tmp_path
    assert captured["kwargs"]["capture_output"] is True
    assert captured["kwargs"]["text"] is True
    assert captured["kwargs"]["timeout"] == 30
    assert captured["kwargs"]["env"]["PYTORCH_ALLOC_CONF"] == "expandable_segments:True"
    assert captured["kwargs"]["env"]["FLASHINFER_TRACE_DIR"] == "/repo"
    assert progress.added == ["Compiling HIP/C++ solution..."]
    assert progress.completed == ["task-1"]


def test_run_compile_phase_filters_benign_rocm_stderr(tmp_path: Path) -> None:
    packager = _FakePackager()
    progress = _FakeProgress()

    def fake_runner(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=args[0],
            returncode=1,
            stdout="",
            stderr=(
                "/opt/amdgpu/share/libdrm/amdgpu.ids: No such file or directory\n"
                "real compiler failure\n"
            ),
        )

    result = cli_compilation.run_compile_phase(
        packager=packager,
        staging_dir=tmp_path,
        compile_timeout=30,
        progress=progress,
        runner=fake_runner,
    )

    assert result.attempted is True
    assert result.succeeded is False
    assert result.returncode == 1
    assert result.filtered_stderr == "real compiler failure\n"
```

- [ ] **Step 2: Run compilation tests to verify they fail**

Run:

```bash
uv run pytest tests/sol_execbench/test_cli_compilation.py -q
```

Expected: FAIL during import because `sol_execbench.cli.compilation` does not exist yet.

- [ ] **Step 3: Create `src/sol_execbench/cli/compilation.py`**

Create `src/sol_execbench/cli/compilation.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""HIP/C++ compile phase helpers for the SOL-ExecBench CLI."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..core.bench.io import flashinfer_safetensors_env
from ..core.bench.stderr import filter_benign_rocm_stderr


@dataclass(frozen=True)
class CompilePhaseResult:
    attempted: bool
    succeeded: bool
    artifact_path: Path | None
    stdout: str
    filtered_stderr: str
    returncode: int


def run_compile_phase(
    *,
    packager: Any,
    staging_dir: Path,
    compile_timeout: int,
    progress: Any,
    env_builder: Callable[
        [Mapping[str, str]], dict[str, str]
    ] = flashinfer_safetensors_env,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> CompilePhaseResult:
    if not packager._is_cpp:
        return CompilePhaseResult(
            attempted=False,
            succeeded=True,
            artifact_path=None,
            stdout="",
            filtered_stderr="",
            returncode=0,
        )

    task = progress.add_task("Compiling HIP/C++ solution...", total=None)
    cmd, artifact_path = packager.compile()
    proc = runner(
        cmd,
        cwd=staging_dir,
        capture_output=True,
        text=True,
        timeout=compile_timeout,
        env=env_builder({**os.environ, "PYTORCH_ALLOC_CONF": "expandable_segments:True"}),
    )
    progress.update(task, completed=True)
    filtered_stderr = filter_benign_rocm_stderr(proc.stderr)
    return CompilePhaseResult(
        attempted=True,
        succeeded=proc.returncode == 0,
        artifact_path=artifact_path,
        stdout=proc.stdout,
        filtered_stderr=filtered_stderr,
        returncode=proc.returncode,
    )
```

- [ ] **Step 4: Update `main.py` to use compilation helper**

In `src/sol_execbench/cli/main.py`, add:

```python
from . import compilation as cli_compilation
```

Remove imports no longer needed by `main.py` solely for compile phase:

```python
import os
from ..core.bench.io import flashinfer_safetensors_env
```

Keep `subprocess` if still used for timeout exception handling and `CompletedProcess` typing in evaluation path.

Replace the current compile phase block:

```python
# Phase 1: Compile (HIP/C++ only)
static_evidence_result: StaticKernelEvidenceSidecar | None = None
if packager._is_cpp:
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Compiling HIP/C++ solution...", total=None)

        cmd, artifact_path = packager.compile()
        proc = subprocess.run(
            cmd,
            cwd=staging_dir,
            capture_output=True,
            text=True,
            timeout=compile_timeout,
            env=flashinfer_safetensors_env(
                {**os.environ, "PYTORCH_ALLOC_CONF": "expandable_segments:True"}
            ),
        )
        progress.update(task, completed=True)

    if proc.returncode != 0:
        console.print("[red]Compilation failed[/red]")
        filtered_stderr = filter_benign_rocm_stderr(proc.stderr)
        if filtered_stderr:
            console.print(filtered_stderr)
        if proc.stdout:
            console.print(proc.stdout)
        packager.close()
        sys.exit(1)

    console.print("[green]Compilation succeeded[/green]")
    filtered_stderr = filter_benign_rocm_stderr(proc.stderr)
    if verbose and filtered_stderr:
        console.print(f"[dim]{filtered_stderr}[/dim]")

    if static_evidence == cli_static_evidence.STATIC_EVIDENCE_AUTO:
        static_evidence_result = cli_static_evidence._collect_static_evidence_for_cli(
            enabled=static_evidence,
            is_cpp=True,
            staging_dir=staging_dir,
            output_file=output_file,
        )
elif static_evidence == cli_static_evidence.STATIC_EVIDENCE_AUTO:
    static_evidence_result = cli_static_evidence._collect_static_evidence_for_cli(
        enabled=static_evidence,
        is_cpp=False,
        staging_dir=staging_dir,
        output_file=output_file,
    )
```

with:

```python
# Phase 1: Compile (HIP/C++ only)
static_evidence_result: StaticKernelEvidenceSidecar | None = None
with Progress(
    SpinnerColumn(),
    TextColumn("[progress.description]{task.description}"),
    console=console,
) as progress:
    compile_result = cli_compilation.run_compile_phase(
        packager=packager,
        staging_dir=staging_dir,
        compile_timeout=compile_timeout,
        progress=progress,
    )

if compile_result.attempted:
    if not compile_result.succeeded:
        console.print("[red]Compilation failed[/red]")
        if compile_result.filtered_stderr:
            console.print(compile_result.filtered_stderr)
        if compile_result.stdout:
            console.print(compile_result.stdout)
        packager.close()
        sys.exit(1)

    console.print("[green]Compilation succeeded[/green]")
    if verbose and compile_result.filtered_stderr:
        console.print(f"[dim]{compile_result.filtered_stderr}[/dim]")

if packager._is_cpp:
    if static_evidence == cli_static_evidence.STATIC_EVIDENCE_AUTO:
        static_evidence_result = cli_static_evidence._collect_static_evidence_for_cli(
            enabled=static_evidence,
            is_cpp=True,
            staging_dir=staging_dir,
            output_file=output_file,
        )
elif static_evidence == cli_static_evidence.STATIC_EVIDENCE_AUTO:
    static_evidence_result = cli_static_evidence._collect_static_evidence_for_cli(
        enabled=static_evidence,
        is_cpp=False,
        staging_dir=staging_dir,
        output_file=output_file,
    )
```

This preserves exit authority in `main.py`, not `compilation.py`.

- [ ] **Step 5: Run compilation and boundary tests**

Run:

```bash
uv run pytest tests/sol_execbench/test_cli_compilation.py tests/sol_execbench/test_cli_module_boundaries.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/sol_execbench/cli/compilation.py src/sol_execbench/cli/main.py tests/sol_execbench/test_cli_compilation.py tests/sol_execbench/test_cli_module_boundaries.py
git commit -s -m "#0 - Extract CLI compile phase helper"
```

---

### Task 4: Run evaluation input/compile regression checks

**Files:**
- No source edits expected unless Ruff reports import-order or lint issues.

- [ ] **Step 1: Run focused new tests**

Run:

```bash
uv run pytest tests/sol_execbench/test_cli_problem_io.py tests/sol_execbench/test_cli_compilation.py tests/sol_execbench/test_cli_module_boundaries.py -q
```

Expected: PASS.

- [ ] **Step 2: Run existing CLI diagnostics and timeout tests**

Run:

```bash
uv run pytest tests/sol_execbench/test_cli_diagnostics.py tests/sol_execbench/test_cli_evaluation_timeout.py tests/sol_execbench/test_cli_environment.py tests/sol_execbench/test_cli_reporting.py -q
```

Expected: PASS.

- [ ] **Step 3: Run focused sidecar tests to cover static/profile/feedback ordering assumptions**

Run:

```bash
uv run pytest tests/sol_execbench/test_cli_profile_sidecars.py tests/sol_execbench/test_cli_static_evidence.py tests/sol_execbench/test_cli_agent_feedback_sidecar.py -q
```

Expected: PASS.

- [ ] **Step 4: Run Ruff on changed modules and tests**

Run:

```bash
uv run --with ruff ruff check src/sol_execbench/cli/main.py src/sol_execbench/cli/problem_io.py src/sol_execbench/cli/compilation.py tests/sol_execbench/test_cli_problem_io.py tests/sol_execbench/test_cli_compilation.py tests/sol_execbench/test_cli_module_boundaries.py
```

Expected: PASS.

- [ ] **Step 5: Commit Ruff-only cleanup if needed**

Only run this if Step 4 required import-order or lint cleanup:

```bash
git add src/sol_execbench/cli/main.py src/sol_execbench/cli/problem_io.py src/sol_execbench/cli/compilation.py tests/sol_execbench/test_cli_problem_io.py tests/sol_execbench/test_cli_compilation.py tests/sol_execbench/test_cli_module_boundaries.py
git commit -s -m "#0 - Verify CLI input compile refactor"
```

---

## Expected End State

- `src/sol_execbench/cli/problem_io.py` owns input path resolution and JSON model loading helpers.
- `src/sol_execbench/cli/compilation.py` owns compile subprocess execution and returns `CompilePhaseResult`.
- `src/sol_execbench/cli/main.py` keeps Click command definition, user-facing orchestration, sidecar ordering, `packager.close()`, and `sys.exit(...)`.
- `_evaluate_cli` is smaller but still behaviorally equivalent for:
  - problem-dir and explicit path handling.
  - config override and `--lock-clocks`.
  - C++ compile success/failure output.
  - non-C++ static evidence unsupported sidecar behavior.
  - evaluation/no-trace/profile/sidecar output ordering.

---

## Self-Review

**Spec coverage:** This plan covers the recommended next refactor from recent-commit analysis: extract problem input/model loading and compile phase without touching the evaluation/no-trace/profile/sidecar orchestration.

**Placeholder scan:** No placeholder items remain. Each task has exact files, code snippets, commands, expected outcomes, and commit messages.

**Type consistency:** `ResolvedProblemInputs`, `CompilePhaseResult`, `resolve_problem_inputs`, and `run_compile_phase` are consistently named across tests, modules, and `main.py` usage.
