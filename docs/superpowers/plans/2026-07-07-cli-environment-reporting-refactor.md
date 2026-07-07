# CLI Environment and Reporting Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split CLI environment snapshot and human-reporting responsibilities out of the large `cli/main.py` and mixed `cli/sidecars.py` modules without changing CLI behavior.

**Architecture:** Keep `src/sol_execbench/cli/main.py` as command orchestration only: argument resolution, compile/evaluate flow, sidecar orchestration, and process exit. Move environment snapshot pathing/writing to `src/sol_execbench/cli/environment.py`, and move Rich trace table rendering to `src/sol_execbench/cli/reporting.py`. Preserve existing output formats, sidecar filenames, nonfatal diagnostic behavior, and exit-code semantics.

**Tech Stack:** Python 3.12, Click, Rich, Pydantic v2 models, pytest, existing SOL ExecBench CLI helpers.

---

## File Structure

**Create:**
- `src/sol_execbench/cli/environment.py`: environment snapshot constants, sidecar path resolution, and nonfatal environment snapshot writing.
- `src/sol_execbench/cli/reporting.py`: human-facing Rich output for trace tables and runtime logs.
- `tests/sol_execbench/test_cli_environment.py`: focused tests for environment snapshot pathing and writing.
- `tests/sol_execbench/test_cli_reporting.py`: focused tests for trace table output behavior.

**Modify:**
- `src/sol_execbench/cli/main.py`: import new modules, remove `_print_traces_table`, and call extracted helpers.
- `src/sol_execbench/cli/sidecars.py`: remove environment snapshot constants/functions and their imports.
- `tests/sol_execbench/test_cli_environment_snapshot.py`: remove tests moved into focused environment/reporting test files, leaving evaluation/profile/static-evidence/agent-feedback coverage.
- `tests/sol_execbench/test_cli_module_boundaries.py`: add assertions that environment and reporting helpers live outside `main.py`, and that environment helpers no longer live in `sidecars.py`.

---

### Task 1: Add module-boundary tests for the next split

**Files:**
- Modify: `tests/sol_execbench/test_cli_module_boundaries.py`

- [ ] **Step 1: Write the failing boundary tests**

Replace the file content with:

```python
from __future__ import annotations

from sol_execbench.cli import evaluation
from sol_execbench.cli import environment
from sol_execbench.cli import main as cli_main
from sol_execbench.cli import reporting
from sol_execbench.cli import sidecars


def test_cli_environment_helpers_live_outside_main_and_sidecars() -> None:
    assert environment._write_environment_snapshot_sidecar is not None
    assert environment._environment_snapshot_sidecar_path is not None
    assert environment.ENV_SNAPSHOT_ENABLE_ENV == "SOLEXECBENCH_ENV_SNAPSHOT"
    assert environment.ENV_SNAPSHOT_PATH_ENV == "SOLEXECBENCH_ENV_SNAPSHOT_PATH"

    for module in (cli_main, sidecars):
        for name in (
            "ENV_SNAPSHOT_ENABLE_ENV",
            "ENV_SNAPSHOT_PATH_ENV",
            "_environment_snapshot_sidecar_path",
            "_write_environment_snapshot_sidecar",
        ):
            assert not hasattr(module, name)


def test_cli_reporting_helpers_live_outside_main() -> None:
    assert reporting.print_traces_table is not None
    assert not hasattr(cli_main, "_print_traces_table")


def test_cli_sidecar_helpers_live_outside_main() -> None:
    assert sidecars._write_profile_sidecar is not None
    assert sidecars._write_profile_summary_sidecar is not None
    assert sidecars._write_static_evidence_sidecar is not None
    assert sidecars._write_agent_feedback_sidecar is not None
    assert sidecars._collect_static_evidence_for_cli is not None

    for name in (
        "_write_profile_sidecar",
        "_write_profile_summary_sidecar",
        "_write_static_evidence_sidecar",
        "_write_agent_feedback_sidecar",
        "_collect_static_evidence_for_cli",
    ):
        assert not hasattr(cli_main, name)


def test_cli_evaluation_helpers_live_outside_main() -> None:
    assert evaluation._write_no_trace_diagnostics_sidecar is not None
    assert evaluation._timeout_output_text is not None
    assert evaluation._run_evaluation_command is not None
    assert evaluation._run_profiled_evaluation is not None

    for name in (
        "_write_no_trace_diagnostics_sidecar",
        "_timeout_output_text",
        "_run_evaluation_command",
        "_run_profiled_evaluation",
    ):
        assert not hasattr(cli_main, name)
```

- [ ] **Step 2: Run boundary test to verify it fails**

Run:

```bash
uv run pytest tests/sol_execbench/test_cli_module_boundaries.py -q
```

Expected: FAIL during import because `sol_execbench.cli.environment` and `sol_execbench.cli.reporting` do not exist yet.

- [ ] **Step 3: Commit**

```bash
git add tests/sol_execbench/test_cli_module_boundaries.py
git commit -s -m "#0 - Add CLI environment reporting boundary tests"
```

---

### Task 2: Extract environment snapshot helpers from `sidecars.py`

**Files:**
- Create: `src/sol_execbench/cli/environment.py`
- Modify: `src/sol_execbench/cli/sidecars.py`
- Modify: `src/sol_execbench/cli/main.py`
- Create: `tests/sol_execbench/test_cli_environment.py`
- Modify: `tests/sol_execbench/test_cli_environment_snapshot.py`

- [ ] **Step 1: Create focused environment tests**

Create `tests/sol_execbench/test_cli_environment.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

from sol_execbench.cli import environment as cli_environment
from sol_execbench.core.environment import (
    EnvironmentEvidenceStatus,
    EnvironmentSnapshot,
)


def _snapshot() -> EnvironmentSnapshot:
    return EnvironmentSnapshot(
        generated_at="2026-05-25T00:00:00+00:00",
        collection_status=EnvironmentEvidenceStatus.AVAILABLE,
    )


def test_environment_snapshot_sidecar_disabled_by_default(
    tmp_path: Path,
    monkeypatch,
):
    output = tmp_path / "trace.jsonl"
    monkeypatch.delenv(cli_environment.ENV_SNAPSHOT_ENABLE_ENV, raising=False)
    monkeypatch.delenv(cli_environment.ENV_SNAPSHOT_PATH_ENV, raising=False)

    written = cli_environment._write_environment_snapshot_sidecar(
        output,
        collector=lambda: _snapshot(),
    )

    assert written is None
    assert not output.with_name("trace.jsonl.environment.json").exists()


def test_environment_snapshot_sidecar_uses_explicit_path(tmp_path: Path, monkeypatch):
    sidecar = tmp_path / "run" / "env.json"
    monkeypatch.setenv(cli_environment.ENV_SNAPSHOT_PATH_ENV, str(sidecar))
    monkeypatch.delenv(cli_environment.ENV_SNAPSHOT_ENABLE_ENV, raising=False)

    written = cli_environment._write_environment_snapshot_sidecar(
        tmp_path / "trace.jsonl",
        collector=lambda: _snapshot(),
    )

    assert written == sidecar
    payload = json.loads(sidecar.read_text())
    assert payload["schema_version"] == "sol_execbench.environment_snapshot.v1"
    assert payload["collection_status"] == "available"


def test_environment_snapshot_sidecar_can_be_derived_from_trace_output(
    tmp_path: Path,
    monkeypatch,
):
    output = tmp_path / "trace.jsonl"
    monkeypatch.setenv(cli_environment.ENV_SNAPSHOT_ENABLE_ENV, "1")
    monkeypatch.delenv(cli_environment.ENV_SNAPSHOT_PATH_ENV, raising=False)

    written = cli_environment._write_environment_snapshot_sidecar(
        output,
        collector=lambda: _snapshot(),
    )

    assert written == tmp_path / "trace.jsonl.environment.json"
    assert written is not None
    assert json.loads(written.read_text())["collection_status"] == "available"


def test_environment_snapshot_request_without_output_path_is_nonfatal(monkeypatch):
    calls = 0

    def collector() -> EnvironmentSnapshot:
        nonlocal calls
        calls += 1
        return _snapshot()

    monkeypatch.setenv(cli_environment.ENV_SNAPSHOT_ENABLE_ENV, "1")
    monkeypatch.delenv(cli_environment.ENV_SNAPSHOT_PATH_ENV, raising=False)

    written = cli_environment._write_environment_snapshot_sidecar(
        None, collector=collector
    )

    assert written is None
    assert calls == 0


def test_environment_snapshot_collection_failure_is_nonfatal(
    tmp_path: Path, monkeypatch
):
    sidecar = tmp_path / "env.json"
    monkeypatch.setenv(cli_environment.ENV_SNAPSHOT_PATH_ENV, str(sidecar))

    def collector() -> EnvironmentSnapshot:
        raise RuntimeError("probe failed")

    written = cli_environment._write_environment_snapshot_sidecar(
        tmp_path / "trace.jsonl",
        collector=collector,
    )

    assert written is None
    assert not sidecar.exists()
```

- [ ] **Step 2: Run focused environment tests to verify they fail**

Run:

```bash
uv run pytest tests/sol_execbench/test_cli_environment.py -q
```

Expected: FAIL during import because `sol_execbench.cli.environment` does not exist yet.

- [ ] **Step 3: Create `src/sol_execbench/cli/environment.py`**

Create `src/sol_execbench/cli/environment.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Environment snapshot helpers for the SOL-ExecBench CLI."""

from __future__ import annotations

import os
from pathlib import Path

from rich.console import Console

from ..core.environment import collect_environment_snapshot
from ..core.runtime_evidence import write_json_payload

console = Console(stderr=True)

ENV_SNAPSHOT_ENABLE_ENV = "SOLEXECBENCH_ENV_SNAPSHOT"
ENV_SNAPSHOT_PATH_ENV = "SOLEXECBENCH_ENV_SNAPSHOT_PATH"


def _environment_snapshot_sidecar_path(output_file: Path | None) -> Path | None:
    """Return the optional environment snapshot sidecar path for this run."""

    explicit = os.environ.get(ENV_SNAPSHOT_PATH_ENV)
    if explicit:
        return Path(explicit)
    if os.environ.get(ENV_SNAPSHOT_ENABLE_ENV) == "1" and output_file is not None:
        return output_file.with_name(f"{output_file.name}.environment.json")
    return None


def _write_environment_snapshot_sidecar(
    output_file: Path | None,
    *,
    collector=collect_environment_snapshot,
) -> Path | None:
    """Write optional environment snapshot sidecar metadata.

    Snapshot evidence is diagnostic only. Collection or serialization failures
    are reported to stderr and never change benchmark correctness status.
    """

    sidecar_path = _environment_snapshot_sidecar_path(output_file)
    if sidecar_path is None:
        if os.environ.get(ENV_SNAPSHOT_ENABLE_ENV) == "1":
            console.print(
                "[yellow]Environment snapshot requested but no output path is available; "
                f"set {ENV_SNAPSHOT_PATH_ENV} or use --output.[/yellow]"
            )
        return None

    try:
        snapshot = collector()
        write_json_payload(sidecar_path, snapshot)
        console.print(f"[green]Saved environment snapshot to {sidecar_path}[/green]")
        return sidecar_path
    except Exception as exc:
        console.print(f"[yellow]Environment snapshot skipped: {exc}[/yellow]")
        return None
```

- [ ] **Step 4: Remove environment code from `sidecars.py`**

In `src/sol_execbench/cli/sidecars.py`, remove:

```python
import os
from ..core.environment import collect_environment_snapshot
ENV_SNAPSHOT_ENABLE_ENV = "SOLEXECBENCH_ENV_SNAPSHOT"
ENV_SNAPSHOT_PATH_ENV = "SOLEXECBENCH_ENV_SNAPSHOT_PATH"
def _environment_snapshot_sidecar_path(...)
def _write_environment_snapshot_sidecar(...)
```

Keep these imports and constants:

```python
from collections.abc import Sequence
from pathlib import Path

from rich.console import Console

from ..core import Solution, Trace
from ..core.bench.agent_feedback import (
    AgentFeedbackArtifactCitation,
    artifact_citation_from_path,
    build_agent_feedback_sidecar,
)
from ..core.bench.profile_summary import (
    ProfileSummaryArtifactCitation,
    build_profile_summary_sidecar,
    profile_summary_artifact_citation_from_path,
)
from ..core.bench.rocm_profiler import Rocprofv3ProfileResult
from ..core.bench.static_kernel_evidence import (
    StaticKernelEvidenceReasonCode,
    StaticKernelEvidenceSidecar,
    StaticKernelEvidenceSourceReference,
    StaticKernelEvidenceWarning,
    build_static_kernel_evidence_failed,
    build_static_kernel_evidence_unsupported,
    collect_static_kernel_artifacts,
    run_static_kernel_extractors,
)
from ..core.dataset.checksums import sha256_file, stable_json_checksum
from ..core.runtime_evidence import write_json_payload
```

- [ ] **Step 5: Update `main.py` to call the new module**

In `src/sol_execbench/cli/main.py`, add the import:

```python
from . import environment as cli_environment
```

Replace:

```python
environment_sidecar_path = cli_sidecars._write_environment_snapshot_sidecar(
    output_file
)
```

with:

```python
environment_sidecar_path = cli_environment._write_environment_snapshot_sidecar(
    output_file
)
```

- [ ] **Step 6: Remove moved tests from `test_cli_environment_snapshot.py`**

Delete these tests and the now-unused `EnvironmentSnapshot` import from `tests/sol_execbench/test_cli_environment_snapshot.py`:

```python
test_environment_snapshot_sidecar_disabled_by_default
test_environment_snapshot_sidecar_uses_explicit_path
test_environment_snapshot_sidecar_can_be_derived_from_trace_output
test_environment_snapshot_request_without_output_path_is_nonfatal
test_environment_snapshot_collection_failure_is_nonfatal
```

Keep the local `_snapshot()` helper because `test_doctor_cli_outputs_json_without_problem_directory` still uses it.

- [ ] **Step 7: Run extracted environment and boundary tests**

Run:

```bash
uv run pytest tests/sol_execbench/test_cli_environment.py tests/sol_execbench/test_cli_module_boundaries.py -q
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/sol_execbench/cli/environment.py src/sol_execbench/cli/main.py src/sol_execbench/cli/sidecars.py tests/sol_execbench/test_cli_environment.py tests/sol_execbench/test_cli_environment_snapshot.py tests/sol_execbench/test_cli_module_boundaries.py
git commit -s -m "#0 - Extract CLI environment snapshot helpers"
```

---

### Task 3: Extract Rich trace-table reporting from `main.py`

**Files:**
- Create: `src/sol_execbench/cli/reporting.py`
- Modify: `src/sol_execbench/cli/main.py`
- Create: `tests/sol_execbench/test_cli_reporting.py`

- [ ] **Step 1: Write focused reporting tests**

Create `tests/sol_execbench/test_cli_reporting.py`:

```python
from __future__ import annotations

from rich.console import Console

from sol_execbench.cli.reporting import print_traces_table
from sol_execbench.core.data.trace import (
    Correctness,
    Environment,
    Evaluation,
    EvaluationStatus,
    Performance,
    Trace,
)
from sol_execbench.core.data.workload import ScalarInput, Workload


def _trace(
    status: EvaluationStatus,
    *,
    log: str = "",
    latency_ms: float | None = 1.25,
    reference_latency_ms: float | None = 2.5,
) -> Trace:
    performance = None
    if latency_ms is not None and reference_latency_ms is not None:
        performance = Performance(
            latency_ms=latency_ms,
            reference_latency_ms=reference_latency_ms,
            speedup_factor=reference_latency_ms / latency_ms,
        )
    return Trace(
        definition="toy",
        solution="candidate",
        workload=Workload(
            uuid="w0",
            axes={"n": 1},
            inputs={"n": ScalarInput(value=1)},
        ),
        evaluation=Evaluation(
            status=status,
            performance=performance,
            correctness=Correctness(
                max_absolute_error=0.0,
                max_relative_error=0.0,
                has_nan=False,
                has_inf=False,
            ),
            environment=Environment(hardware="AMD gfx1200", libs={"hip": "7.0"}),
            timestamp="2026-06-16T00:00:00Z",
            log=log,
        ),
    )


def test_print_traces_table_reports_pass_count_and_speedup() -> None:
    console = Console(record=True, width=120)

    print_traces_table(
        [
            _trace(EvaluationStatus.PASSED),
            _trace(EvaluationStatus.INCORRECT_NUMERICAL),
        ],
        console=console,
    )

    output = console.export_text()
    assert "Evaluation Results" in output
    assert "PASSED" in output
    assert "INCORRECT_NUMERICAL" in output
    assert "2.00x" in output
    assert "1/2 workloads passed" in output


def test_print_traces_table_prints_runtime_logs_for_runtime_failures() -> None:
    console = Console(record=True, width=120)

    print_traces_table(
        [
            _trace(EvaluationStatus.RUNTIME_ERROR, log="kernel launch failed\n"),
            _trace(EvaluationStatus.INCORRECT_NUMERICAL, log="not runtime fatal\n"),
        ],
        console=console,
    )

    output = console.export_text()
    assert "Runtime logs (1)" in output
    assert "Workload 0" in output
    assert "RUNTIME_ERROR" in output
    assert "kernel launch failed" in output
    assert "not runtime fatal" not in output


def test_print_traces_table_handles_missing_evaluation() -> None:
    console = Console(record=True, width=120)
    trace = _trace(EvaluationStatus.PASSED)
    trace = trace.model_copy(update={"evaluation": None})

    print_traces_table([trace], console=console)

    output = console.export_text()
    assert "no evaluation" in output
    assert "0/1 workloads passed" in output
```

- [ ] **Step 2: Run focused reporting tests to verify they fail**

Run:

```bash
uv run pytest tests/sol_execbench/test_cli_reporting.py -q
```

Expected: FAIL during import because `sol_execbench.cli.reporting` does not exist yet.

- [ ] **Step 3: Create `src/sol_execbench/cli/reporting.py`**

Create `src/sol_execbench/cli/reporting.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Human-facing CLI reporting helpers."""

from __future__ import annotations

from rich.console import Console
from rich.table import Table

from ..core import EvaluationStatus, Trace

console = Console(stderr=True)


def print_traces_table(
    traces: list[Trace],
    *,
    console: Console = console,
) -> None:
    """Print a rich table summarizing evaluation traces."""

    table = Table(title="Evaluation Results", show_lines=True)
    table.add_column("#", style="dim", width=4)
    table.add_column("Status", width=22)
    table.add_column("Latency (ms)", justify="right", width=14)
    table.add_column("Ref (ms)", justify="right", width=14)
    table.add_column("Speedup", justify="right", width=10)
    table.add_column("Max Abs Err", justify="right", width=14)
    table.add_column("Max Rel Err", justify="right", width=14)

    passed = 0
    total = len(traces)
    for i, trace in enumerate(traces):
        ev = trace.evaluation
        if ev is None:
            table.add_row(str(i), "[dim]no evaluation[/dim]", "", "", "", "", "")
            continue

        status = ev.status.value
        if ev.status == EvaluationStatus.PASSED:
            status_str = f"[green]{status}[/green]"
            passed += 1
        elif ev.status == EvaluationStatus.INCORRECT_NUMERICAL:
            status_str = f"[yellow]{status}[/yellow]"
        else:
            status_str = f"[red]{status}[/red]"

        latency = ""
        ref_latency = ""
        speedup = ""
        if ev.performance:
            latency = f"{ev.performance.latency_ms:.3f}"
            ref_latency = f"{ev.performance.reference_latency_ms:.3f}"
            speedup = f"{ev.performance.speedup_factor:.2f}x"

        abs_err = ""
        rel_err = ""
        if ev.correctness:
            if ev.correctness.has_nan:
                abs_err = "NaN"
                rel_err = "NaN"
            elif ev.correctness.has_inf:
                abs_err = "Inf"
                rel_err = "Inf"
            else:
                abs_err = f"{ev.correctness.max_absolute_error:.2e}"
                rel_err = f"{ev.correctness.max_relative_error:.2e}"

        table.add_row(
            str(i), status_str, latency, ref_latency, speedup, abs_err, rel_err
        )

    console.print(table)
    console.print(f"\n[bold]{passed}/{total}[/bold] workloads passed")

    error_logs = []
    for i, trace in enumerate(traces):
        ev = trace.evaluation
        if ev is None:
            continue
        if (
            ev.status
            not in (EvaluationStatus.PASSED, EvaluationStatus.INCORRECT_NUMERICAL)
            and ev.log
        ):
            error_logs.append((i, ev.status.value, ev.log))

    if error_logs:
        console.print(f"\n[bold red]Runtime logs ({len(error_logs)}):[/bold red]")
        for idx, status, log in error_logs:
            console.print(f"\n[bold]Workload {idx}[/bold] ([red]{status}[/red]):")
            console.print(log.rstrip())
```

- [ ] **Step 4: Update `main.py` imports**

In `src/sol_execbench/cli/main.py`, add:

```python
from . import reporting as cli_reporting
```

Remove:

```python
from rich.table import Table
```

Remove `EvaluationStatus` from the `from ..core import (...)` import list only if it is no longer used elsewhere in `main.py`. Keep `Trace` only if `_print_traces_table` has not yet been deleted; after deletion, remove `Trace` too if unused.

- [ ] **Step 5: Remove `_print_traces_table` from `main.py`**

Delete the entire function currently beginning with:

```python
def _print_traces_table(traces: list[Trace]) -> None:
```

and ending after the runtime log printing loop.

- [ ] **Step 6: Update the CLI call site**

Replace:

```python
_print_traces_table(traces)
```

with:

```python
cli_reporting.print_traces_table(traces)
```

- [ ] **Step 7: Run reporting and boundary tests**

Run:

```bash
uv run pytest tests/sol_execbench/test_cli_reporting.py tests/sol_execbench/test_cli_module_boundaries.py -q
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/sol_execbench/cli/reporting.py src/sol_execbench/cli/main.py tests/sol_execbench/test_cli_reporting.py tests/sol_execbench/test_cli_module_boundaries.py
git commit -s -m "#0 - Extract CLI trace reporting"
```

---

### Task 4: Trim the mixed CLI test file after extraction

**Files:**
- Modify: `tests/sol_execbench/test_cli_environment_snapshot.py`

- [ ] **Step 1: Ensure the mixed test file no longer imports moved modules only for moved tests**

In `tests/sol_execbench/test_cli_environment_snapshot.py`, remove imports that are unused after Task 2 and Task 3. The top of the file should still include imports needed by the remaining tests:

```python
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from click.testing import CliRunner

from sol_execbench.cli import evaluation as cli_evaluation
from sol_execbench.cli import main as cli_main
from sol_execbench.cli import sidecars as cli_sidecars
from sol_execbench.cli.main import cli
from sol_execbench.core.bench.rocm_profiler import Rocprofv3ProfileResult
from sol_execbench.core.bench.rocm_profiler import Rocprofv3ProfileArtifact
from sol_execbench.core.bench.static_kernel_evidence import (
    StaticKernelEvidenceArtifact,
    StaticKernelEvidenceReasonCode,
    StaticKernelEvidenceStatus,
    build_static_kernel_evidence_sidecar,
)
from sol_execbench.core.data.solution import (
    BuildSpec,
    Solution,
    SourceFile,
    SupportedHardware,
    SupportedLanguages,
)
from sol_execbench.core.data.trace import (
    Environment,
    Evaluation,
    EvaluationStatus,
    Trace,
)
from sol_execbench.core.data.workload import ScalarInput
from sol_execbench.core.data.workload import Workload
from sol_execbench.core.environment import (
    EnvironmentCheckResult,
    EnvironmentDiagnostics,
    EnvironmentEvidenceStatus,
    EnvironmentSnapshot,
)
```

- [ ] **Step 2: Rename file if the remaining scope is no longer environment-specific**

If the remaining tests are primarily subprocess diagnostics, profiling sidecars, static evidence, agent feedback, and doctor CLI behavior, rename:

```bash
git mv tests/sol_execbench/test_cli_environment_snapshot.py tests/sol_execbench/test_cli_diagnostics.py
```

If `git mv` is not available in the execution environment, use:

```bash
mv tests/sol_execbench/test_cli_environment_snapshot.py tests/sol_execbench/test_cli_diagnostics.py
git add tests/sol_execbench/test_cli_environment_snapshot.py tests/sol_execbench/test_cli_diagnostics.py
```

- [ ] **Step 3: Run CLI diagnostics tests**

Run:

```bash
uv run pytest tests/sol_execbench/test_cli_diagnostics.py tests/sol_execbench/test_cli_environment.py tests/sol_execbench/test_cli_reporting.py tests/sol_execbench/test_cli_module_boundaries.py -q
```

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/sol_execbench/test_cli_diagnostics.py tests/sol_execbench/test_cli_environment.py tests/sol_execbench/test_cli_reporting.py tests/sol_execbench/test_cli_module_boundaries.py
git commit -s -m "#0 - Split CLI diagnostics tests"
```

---

### Task 5: Run targeted regression checks

**Files:**
- No source edits expected.

- [ ] **Step 1: Run focused CLI test set**

Run:

```bash
uv run pytest tests/sol_execbench/test_cli_diagnostics.py tests/sol_execbench/test_cli_environment.py tests/sol_execbench/test_cli_reporting.py tests/sol_execbench/test_cli_module_boundaries.py tests/sol_execbench/test_cli_evaluation_timeout.py -q
```

Expected: PASS.

- [ ] **Step 2: Run contract tests that cover sidecar authority expectations**

Run:

```bash
uv run pytest tests/sol_execbench/test_contract.py tests/sol_execbench/test_agent_feedback.py tests/sol_execbench/test_profile_summary.py -q
```

Expected: PASS.

- [ ] **Step 3: Run lint on changed Python files**

Run:

```bash
uv run --with ruff ruff check src/sol_execbench/cli/main.py src/sol_execbench/cli/sidecars.py src/sol_execbench/cli/environment.py src/sol_execbench/cli/reporting.py tests/sol_execbench/test_cli_diagnostics.py tests/sol_execbench/test_cli_environment.py tests/sol_execbench/test_cli_reporting.py tests/sol_execbench/test_cli_module_boundaries.py
```

Expected: PASS with no Ruff findings.

- [ ] **Step 4: Commit verification-only cleanup if Ruff required import ordering changes**

Only run this if Step 3 required import-order or formatting edits:

```bash
git add src/sol_execbench/cli/main.py src/sol_execbench/cli/sidecars.py src/sol_execbench/cli/environment.py src/sol_execbench/cli/reporting.py tests/sol_execbench/test_cli_diagnostics.py tests/sol_execbench/test_cli_environment.py tests/sol_execbench/test_cli_reporting.py tests/sol_execbench/test_cli_module_boundaries.py
git commit -s -m "#0 - Verify CLI environment reporting refactor"
```

---

## Expected End State

- `src/sol_execbench/cli/main.py` no longer owns environment snapshot pathing or Rich table rendering.
- `src/sol_execbench/cli/sidecars.py` only owns diagnostic sidecars tied to profile summary, static evidence, and agent feedback.
- `src/sol_execbench/cli/environment.py` owns `SOLEXECBENCH_ENV_SNAPSHOT` and `SOLEXECBENCH_ENV_SNAPSHOT_PATH` behavior.
- `src/sol_execbench/cli/reporting.py` owns trace result rendering.
- CLI behavior remains unchanged:
  - environment snapshot sidecars remain opt-in.
  - explicit environment snapshot path still wins over derived trace path.
  - missing `--output` remains nonfatal when environment snapshot collection is requested through `SOLEXECBENCH_ENV_SNAPSHOT=1`.
  - JSON output mode still prints trace JSONL to stdout.
  - non-JSON output mode still prints the same Rich results table and runtime logs.

---

## Self-Review

**Spec coverage:** The plan covers the recommended next refactor area from the commit analysis: CLI environment snapshot extraction, CLI reporting extraction, and test-file split. It explicitly avoids GSD and writes only to `docs/superpowers/plans/`.

**Placeholder scan:** The plan contains no TBD/TODO/fill-later placeholders. Each implementation task includes concrete files, code snippets, commands, and expected results.

**Type consistency:** New modules are imported as `cli_environment` and `cli_reporting` in `main.py`; tests import `sol_execbench.cli.environment` and `sol_execbench.cli.reporting`. Function names are stable across tasks: `_write_environment_snapshot_sidecar`, `_environment_snapshot_sidecar_path`, and `print_traces_table`.
