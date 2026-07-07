# CLI Command Surface Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split non-evaluation CLI subcommands out of `src/sol_execbench/cli/main.py` so the root CLI module only owns evaluator orchestration and command dispatch.

**Architecture:** Move thin Click command groups into focused modules by command domain: metadata commands into `cli/metadata.py`, measured baseline export into `cli/baseline.py`, and dataset migration commands into `cli/dataset.py`. Keep command object names private and imported by `main.py` for root dispatch, preserving existing public CLI syntax and JSON output behavior.

**Tech Stack:** Python 3.12, Click, Rich, pytest, Pydantic v2 model serialization, existing SOL ExecBench CLI command pattern.

---

## File Structure

**Create:**
- `src/sol_execbench/cli/metadata.py`: `contract`, `doctor`, and `toolchain` command objects.
- `src/sol_execbench/cli/baseline.py`: `baseline` command group and `baseline export` command object.
- `src/sol_execbench/cli/dataset.py`: `dataset` command group and migration command objects.
- `tests/sol_execbench/test_cli_metadata_commands.py`: focused command tests for `contract`, `doctor`, and `toolchain`.
- `tests/sol_execbench/test_cli_baseline_commands.py`: focused command tests for `baseline export`.
- `tests/sol_execbench/test_cli_dataset_commands.py`: focused command tests for dataset migration commands.

**Modify:**
- `src/sol_execbench/cli/main.py`: remove non-evaluation command implementations, import new command modules, and dispatch to the moved command objects.
- `tests/sol_execbench/test_cli_module_boundaries.py`: assert command objects live outside `main.py`.
- `tests/sol_execbench/test_cli_diagnostics.py`: remove doctor CLI tests after moving them to `test_cli_metadata_commands.py`.

---

## Behavioral Constraints

- `sol-execbench contract --json` output stays byte-compatible apart from key order already controlled by `sort_keys=True`.
- `sol-execbench doctor --json` output remains GPU-free and still rejects non-JSON mode.
- `sol-execbench toolchain --json` and `sol-execbench toolchain --json --list-registry` keep current JSON behavior.
- `sol-execbench baseline export ...` keeps current file output and optional JSON stdout behavior.
- `sol-execbench dataset migrate-sol ...` and `sol-execbench dataset migrate-flashinfer ...` keep current manifest writing and optional JSON stdout behavior.
- The root evaluator path and `_evaluate_cli` are not refactored in this plan.
- No GSD commands or `.planning/` workflow updates are part of this plan.

---

### Task 1: Add command module boundary tests

**Files:**
- Modify: `tests/sol_execbench/test_cli_module_boundaries.py`

- [ ] **Step 1: Extend boundary tests**

Append these tests to `tests/sol_execbench/test_cli_module_boundaries.py`:

```python
def test_cli_metadata_commands_live_outside_main() -> None:
    from sol_execbench.cli import metadata

    assert metadata._contract_cli is not None
    assert metadata._doctor_cli is not None
    assert metadata._toolchain_cli is not None

    for name in (
        "_contract_cli",
        "_doctor_cli",
        "_toolchain_cli",
    ):
        assert not hasattr(cli_main, name)


def test_cli_baseline_commands_live_outside_main() -> None:
    from sol_execbench.cli import baseline

    assert baseline._baseline_cli is not None
    assert baseline._baseline_export_cli is not None

    for name in (
        "_baseline_cli",
        "_baseline_export_cli",
    ):
        assert not hasattr(cli_main, name)


def test_cli_dataset_commands_live_outside_main() -> None:
    from sol_execbench.cli import dataset

    assert dataset._dataset_cli is not None
    assert dataset._dataset_migrate_sol_cli is not None
    assert dataset._dataset_migrate_flashinfer_cli is not None

    for name in (
        "_dataset_cli",
        "_dataset_migrate_sol_cli",
        "_dataset_migrate_flashinfer_cli",
    ):
        assert not hasattr(cli_main, name)
```

- [ ] **Step 2: Run boundary tests to verify they fail**

Run:

```bash
uv run pytest tests/sol_execbench/test_cli_module_boundaries.py -q
```

Expected: FAIL because `sol_execbench.cli.metadata`, `sol_execbench.cli.baseline`, and `sol_execbench.cli.dataset` do not exist yet.

- [ ] **Step 3: Commit**

```bash
git add tests/sol_execbench/test_cli_module_boundaries.py
git commit -s -m "#0 - Add CLI command module boundary tests"
```

---

### Task 2: Extract metadata commands

**Files:**
- Create: `src/sol_execbench/cli/metadata.py`
- Modify: `src/sol_execbench/cli/main.py`
- Create: `tests/sol_execbench/test_cli_metadata_commands.py`
- Modify: `tests/sol_execbench/test_cli_diagnostics.py`
- Modify: `tests/sol_execbench/test_cli_module_boundaries.py`

- [ ] **Step 1: Create focused metadata command tests**

Create `tests/sol_execbench/test_cli_metadata_commands.py`:

```python
from __future__ import annotations

import json

from click.testing import CliRunner

from sol_execbench.cli import metadata as cli_metadata
from sol_execbench.cli.main import cli
from sol_execbench.core.environment import (
    EnvironmentCheckResult,
    EnvironmentDiagnostics,
    EnvironmentEvidenceStatus,
    EnvironmentSnapshot,
)
from sol_execbench.core.toolchain import (
    ToolchainArtifactType,
    ToolchainCapability,
    ToolchainEvidenceLevel,
    ToolchainRoutingReport,
    ToolchainRoutingRequest,
)


def _snapshot() -> EnvironmentSnapshot:
    return EnvironmentSnapshot(
        generated_at="2026-05-25T00:00:00+00:00",
        collection_status=EnvironmentEvidenceStatus.AVAILABLE,
    )


def test_contract_cli_outputs_json_without_problem_directory() -> None:
    result = CliRunner().invoke(cli, ["contract", "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["schema_version"].startswith("sol_execbench.evaluator_contract.")
    assert "capabilities" in payload


def test_contract_cli_rejects_non_json_mode() -> None:
    result = CliRunner().invoke(cli, ["contract"])

    assert result.exit_code != 0
    assert "Only --json output is supported for contract" in result.output


def test_doctor_cli_outputs_json_without_problem_directory(monkeypatch) -> None:
    diagnostics = EnvironmentDiagnostics(
        generated_at="2026-05-25T00:00:00+00:00",
        status=EnvironmentEvidenceStatus.AVAILABLE,
        snapshot=_snapshot(),
        checks=[
            EnvironmentCheckResult(
                name="pytorch_rocm_runtime",
                status=EnvironmentEvidenceStatus.AVAILABLE,
                message="ok",
            )
        ],
    )
    monkeypatch.setattr(
        cli_metadata, "build_environment_diagnostics", lambda: diagnostics
    )

    result = CliRunner().invoke(cli, ["doctor", "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["schema_version"] == "sol_execbench.environment_diagnostics.v1"
    assert (
        payload["snapshot"]["schema_version"] == "sol_execbench.environment_snapshot.v1"
    )
    assert payload["checks"][0]["name"] == "pytorch_rocm_runtime"


def test_doctor_cli_rejects_non_json_mode() -> None:
    result = CliRunner().invoke(cli, ["doctor"])

    assert result.exit_code != 0
    assert "Only --json output is supported for doctor" in result.output


def test_toolchain_cli_outputs_routing_json(monkeypatch) -> None:
    def fake_report(request: ToolchainRoutingRequest) -> ToolchainRoutingReport:
        assert request.evidence_level == ToolchainEvidenceLevel.PROFILING
        assert request.artifact_type == ToolchainArtifactType.EXECUTABLE_RUN
        assert request.gpu_architecture == "gfx1200"
        return ToolchainRoutingReport(
            request=request,
            selected_tool="rocprofv3",
            supported=True,
            reason="supported",
            evidence_level=ToolchainEvidenceLevel.PROFILING,
            artifact_type=ToolchainArtifactType.EXECUTABLE_RUN,
        )

    monkeypatch.setattr(cli_metadata, "build_toolchain_routing_report", fake_report)

    result = CliRunner().invoke(
        cli,
        [
            "toolchain",
            "--json",
            "--gpu-arch",
            "gfx1200",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["selected_tool"] == "rocprofv3"
    assert payload["supported"] is True


def test_toolchain_cli_lists_registry_json(monkeypatch) -> None:
    monkeypatch.setattr(
        cli_metadata,
        "default_toolchain_registry",
        lambda: [
            ToolchainCapability(
                name="rocprofv3",
                evidence_levels=[ToolchainEvidenceLevel.PROFILING],
                artifact_types=[ToolchainArtifactType.EXECUTABLE_RUN],
                supported=True,
                notes="profiling evidence",
            )
        ],
    )

    result = CliRunner().invoke(cli, ["toolchain", "--json", "--list-registry"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload[0]["name"] == "rocprofv3"
    assert payload[0]["supported"] is True


def test_toolchain_cli_rejects_non_json_mode() -> None:
    result = CliRunner().invoke(cli, ["toolchain"])

    assert result.exit_code != 0
    assert "Only --json output is supported for toolchain" in result.output
```

- [ ] **Step 2: Run metadata tests to verify they fail**

Run:

```bash
uv run pytest tests/sol_execbench/test_cli_metadata_commands.py -q
```

Expected: FAIL during import because `sol_execbench.cli.metadata` does not exist yet.

- [ ] **Step 3: Create `src/sol_execbench/cli/metadata.py`**

Create `src/sol_execbench/cli/metadata.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""GPU-free metadata commands for the SOL-ExecBench CLI."""

from __future__ import annotations

import json

import click

from ..core.data.contract import build_evaluator_contract
from ..core.environment import build_environment_diagnostics
from ..core.toolchain import (
    ToolchainArtifactType,
    ToolchainEvidenceLevel,
    ToolchainRoutingRequest,
    build_toolchain_routing_report,
    default_toolchain_registry,
)


@click.command("contract", context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--json", "json_output", is_flag=True, help="Print contract JSON")
def _contract_cli(json_output: bool) -> None:
    """Print the GPU-free evaluator compatibility contract."""

    if not json_output:
        raise click.ClickException("Only --json output is supported for contract")
    payload = build_evaluator_contract().model_dump(mode="json")
    click.echo(json.dumps(payload, sort_keys=True))


@click.command("doctor", context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--json", "json_output", is_flag=True, help="Print diagnostics JSON")
def _doctor_cli(json_output: bool) -> None:
    """Print ROCm environment diagnostics."""

    if not json_output:
        raise click.ClickException("Only --json output is supported for doctor")
    payload = build_environment_diagnostics().model_dump(mode="json")
    click.echo(json.dumps(payload, sort_keys=True))


@click.command("toolchain", context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--json", "json_output", is_flag=True, help="Print routing JSON")
@click.option(
    "--evidence-level",
    type=click.Choice([level.value for level in ToolchainEvidenceLevel]),
    default=ToolchainEvidenceLevel.PROFILING.value,
    show_default=True,
    help="Evidence level to route",
)
@click.option(
    "--artifact-type",
    type=click.Choice([artifact.value for artifact in ToolchainArtifactType]),
    default=ToolchainArtifactType.EXECUTABLE_RUN.value,
    show_default=True,
    help="Artifact type to route",
)
@click.option("--gpu-arch", "gpu_architecture", help="GPU architecture such as gfx1200")
@click.option("--hardware-generation", help="Hardware generation such as RDNA 4")
@click.option("--rocm-version", help="ROCm version such as 7.0")
@click.option(
    "--list-registry",
    is_flag=True,
    help="Print registry entries instead of a routing decision",
)
def _toolchain_cli(
    json_output: bool,
    evidence_level: str,
    artifact_type: str,
    gpu_architecture: str | None,
    hardware_generation: str | None,
    rocm_version: str | None,
    list_registry: bool,
) -> None:
    """Print ROCm toolchain routing diagnostics."""

    if not json_output:
        raise click.ClickException("Only --json output is supported for toolchain")
    if list_registry:
        payload = [
            capability.model_dump(mode="json")
            for capability in default_toolchain_registry()
        ]
        click.echo(json.dumps(payload, sort_keys=True))
        return
    request = ToolchainRoutingRequest(
        evidence_level=ToolchainEvidenceLevel(evidence_level),
        artifact_type=ToolchainArtifactType(artifact_type),
        gpu_architecture=gpu_architecture,
        hardware_generation=hardware_generation,
        rocm_version=rocm_version,
    )
    payload = build_toolchain_routing_report(request).model_dump(mode="json")
    click.echo(json.dumps(payload, sort_keys=True))
```

- [ ] **Step 4: Update `main.py` metadata dispatch**

In `src/sol_execbench/cli/main.py`, add:

```python
from . import metadata as cli_metadata
```

Remove these imports if they become unused:

```python
from ..core.data.contract import build_evaluator_contract
from ..core.environment import build_environment_diagnostics
from ..core.toolchain import (
    ToolchainArtifactType,
    ToolchainEvidenceLevel,
    ToolchainRoutingRequest,
    build_toolchain_routing_report,
    default_toolchain_registry,
)
```

Delete the `_contract_cli`, `_doctor_cli`, and `_toolchain_cli` function definitions from `main.py`.

Replace command dispatch references:

```python
return _contract_cli.main(
```

with:

```python
return cli_metadata._contract_cli.main(
```

Replace:

```python
return _doctor_cli.main(
```

with:

```python
return cli_metadata._doctor_cli.main(
```

Replace:

```python
return _toolchain_cli.main(
```

with:

```python
return cli_metadata._toolchain_cli.main(
```

- [ ] **Step 5: Remove moved doctor tests from diagnostics test file**

Delete these tests from `tests/sol_execbench/test_cli_diagnostics.py` because `tests/sol_execbench/test_cli_metadata_commands.py` now owns them:

```python
test_doctor_cli_outputs_json_without_problem_directory
test_doctor_cli_rejects_non_json_mode
```

If `_snapshot()` in `test_cli_diagnostics.py` becomes unused after deleting those tests, remove `_snapshot()` and the environment diagnostics imports used only by `_snapshot()`.

- [ ] **Step 6: Run metadata and boundary tests**

Run:

```bash
uv run pytest tests/sol_execbench/test_cli_metadata_commands.py tests/sol_execbench/test_cli_module_boundaries.py -q
```

Expected: metadata tests pass, but boundary tests may still fail on missing `baseline` and `dataset` modules until Tasks 3 and 4. If boundary tests fail only because `baseline` or `dataset` does not exist yet, run this accepted Task 2 verification instead:

```bash
uv run pytest tests/sol_execbench/test_cli_metadata_commands.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/sol_execbench/cli/metadata.py src/sol_execbench/cli/main.py tests/sol_execbench/test_cli_metadata_commands.py tests/sol_execbench/test_cli_diagnostics.py tests/sol_execbench/test_cli_module_boundaries.py
git commit -s -m "#0 - Extract CLI metadata commands"
```

---

### Task 3: Extract baseline command group

**Files:**
- Create: `src/sol_execbench/cli/baseline.py`
- Modify: `src/sol_execbench/cli/main.py`
- Create: `tests/sol_execbench/test_cli_baseline_commands.py`
- Modify: `tests/sol_execbench/test_cli_module_boundaries.py`

- [ ] **Step 1: Create baseline command tests**

Create `tests/sol_execbench/test_cli_baseline_commands.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from sol_execbench.cli import baseline as cli_baseline
from sol_execbench.cli.main import cli


def test_baseline_export_cli_writes_registry_and_prints_human_message(
    tmp_path: Path,
    monkeypatch,
) -> None:
    trace = tmp_path / "trace.jsonl"
    output = tmp_path / "baseline.json"
    trace.write_text('{"definition":"toy"}\n')

    calls: list[dict[str, object]] = []

    def fake_export(**kwargs):
        calls.append(kwargs)
        output.write_text('{"schema_version":"hip_baseline_registry.v1"}\n')
        return {"schema_version": "hip_baseline_registry.v1", "entries": []}

    monkeypatch.setattr(cli_baseline, "export_hip_baseline_registry", fake_export)

    result = CliRunner().invoke(
        cli,
        [
            "baseline",
            "export",
            "--trace",
            str(trace),
            "--output",
            str(output),
            "--target-id",
            "gemm",
            "--sol-version",
            "v1.42",
            "--timing-policy",
            "latency_ms",
        ],
    )

    assert result.exit_code == 0, result.output
    assert calls == [
        {
            "trace_path": trace,
            "output_path": output,
            "target_id": "gemm",
            "sol_version": "v1.42",
            "timing_policy": "latency_ms",
        }
    ]
    assert "Wrote measured baseline registry" in result.output


def test_baseline_export_cli_prints_json_when_requested(
    tmp_path: Path,
    monkeypatch,
) -> None:
    trace = tmp_path / "trace.jsonl"
    output = tmp_path / "baseline.json"
    trace.write_text('{"definition":"toy"}\n')

    monkeypatch.setattr(
        cli_baseline,
        "export_hip_baseline_registry",
        lambda **kwargs: {
            "schema_version": "hip_baseline_registry.v1",
            "target_id": kwargs["target_id"],
        },
    )

    result = CliRunner().invoke(
        cli,
        [
            "baseline",
            "export",
            "--trace",
            str(trace),
            "--output",
            str(output),
            "--target-id",
            "gemm",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload == {
        "schema_version": "hip_baseline_registry.v1",
        "target_id": "gemm",
    }
```

- [ ] **Step 2: Run baseline tests to verify they fail**

Run:

```bash
uv run pytest tests/sol_execbench/test_cli_baseline_commands.py -q
```

Expected: FAIL during import because `sol_execbench.cli.baseline` does not exist yet.

- [ ] **Step 3: Create `src/sol_execbench/cli/baseline.py`**

Create `src/sol_execbench/cli/baseline.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Measured baseline export commands for the SOL-ExecBench CLI."""

from __future__ import annotations

import json
from pathlib import Path

import click
from rich.console import Console

from ..core.baseline_export import export_hip_baseline_registry

console = Console(stderr=True)


@click.group("baseline", context_settings={"help_option_names": ["-h", "--help"]})
def _baseline_cli() -> None:
    """Measured baseline export utilities."""


@_baseline_cli.command(
    "export", context_settings={"help_option_names": ["-h", "--help"]}
)
@click.option(
    "--trace",
    "trace_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="SOL trace JSONL produced by sol-execbench.",
)
@click.option(
    "--output",
    "output_path",
    required=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Write HIP baseline_registry.v1 JSON here.",
)
@click.option("--target-id", required=True, help="HIP target id, such as gemm.")
@click.option(
    "--sol-version",
    default="unknown",
    show_default=True,
    help="SOL version or source revision to record in baseline provenance.",
)
@click.option(
    "--timing-policy",
    default="latency_ms",
    show_default=True,
    help="Timing policy label to record in baseline provenance.",
)
@click.option("--json", "json_output", is_flag=True, help="Print registry JSON")
def _baseline_export_cli(
    trace_path: Path,
    output_path: Path,
    target_id: str,
    sol_version: str,
    timing_policy: str,
    json_output: bool,
) -> None:
    """Export a HIP measured baseline registry from a SOL trace JSONL file."""

    registry = export_hip_baseline_registry(
        trace_path=trace_path,
        output_path=output_path,
        target_id=target_id,
        sol_version=sol_version,
        timing_policy=timing_policy,
    )
    if json_output:
        click.echo(json.dumps(registry, sort_keys=True))
    else:
        console.print(
            f"[green]Wrote measured baseline registry to {output_path}[/green]"
        )
```

- [ ] **Step 4: Update `main.py` baseline dispatch**

In `src/sol_execbench/cli/main.py`, add:

```python
from . import baseline as cli_baseline
```

Remove this import if it becomes unused:

```python
from ..core.baseline_export import export_hip_baseline_registry
```

Delete `_baseline_cli` and `_baseline_export_cli` from `main.py`.

Replace:

```python
return _baseline_cli.main(
```

with:

```python
return cli_baseline._baseline_cli.main(
```

- [ ] **Step 5: Run baseline and boundary tests**

Run:

```bash
uv run pytest tests/sol_execbench/test_cli_baseline_commands.py tests/sol_execbench/test_cli_module_boundaries.py -q
```

Expected: baseline tests pass, but boundary tests may still fail on missing `dataset` module until Task 4. If boundary tests fail only because `dataset` does not exist yet, run this accepted Task 3 verification instead:

```bash
uv run pytest tests/sol_execbench/test_cli_baseline_commands.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/sol_execbench/cli/baseline.py src/sol_execbench/cli/main.py tests/sol_execbench/test_cli_baseline_commands.py tests/sol_execbench/test_cli_module_boundaries.py
git commit -s -m "#0 - Extract CLI baseline commands"
```

---

### Task 4: Extract dataset command group

**Files:**
- Create: `src/sol_execbench/cli/dataset.py`
- Modify: `src/sol_execbench/cli/main.py`
- Create: `tests/sol_execbench/test_cli_dataset_commands.py`
- Modify: `tests/sol_execbench/test_cli_module_boundaries.py`

- [ ] **Step 1: Create dataset command tests**

Create `tests/sol_execbench/test_cli_dataset_commands.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from click.testing import CliRunner

from sol_execbench.cli import dataset as cli_dataset
from sol_execbench.cli.main import cli


@dataclass
class _FakeDenominators:
    migrated_problems: int = 2
    discovered_problems: int = 3
    blockers: int = 1


@dataclass
class _FakeManifest:
    denominators: _FakeDenominators

    def to_json(self) -> str:
        return '{"schema_version":"fake.migration_manifest.v1"}\n'


def test_dataset_migrate_sol_writes_manifest_and_prints_summary(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    manifest = _FakeManifest(denominators=_FakeDenominators())
    calls: list[dict[str, object]] = []

    def fake_migrate_sol_execbench(*args, **kwargs):
        calls.append(
            {
                "source_root": args[0],
                "output_root": args[1],
                "categories": kwargs["categories"],
                "source_revision": kwargs["source_revision"],
            }
        )
        return manifest

    def fake_write_migration_manifest(manifest_arg, target):
        assert manifest_arg is manifest
        Path(target).write_text(manifest.to_json())

    monkeypatch.setattr(
        cli_dataset, "migrate_sol_execbench", fake_migrate_sol_execbench
    )
    monkeypatch.setattr(
        cli_dataset, "write_migration_manifest", fake_write_migration_manifest
    )

    result = CliRunner().invoke(
        cli,
        [
            "dataset",
            "migrate-sol",
            str(source),
            str(output),
            "--category",
            "level1",
            "--category",
            "level2",
            "--source-revision",
            "abc123",
        ],
    )

    assert result.exit_code == 0, result.output
    assert calls == [
        {
            "source_root": source,
            "output_root": output,
            "categories": ("level1", "level2"),
            "source_revision": "abc123",
        }
    ]
    assert (output / "migration-manifest.json").read_text() == manifest.to_json()
    assert "Problems:" in result.output
    assert "2/3 migrated" in result.output


def test_dataset_migrate_sol_prints_json_when_requested(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    explicit_manifest = tmp_path / "manifest.json"
    source.mkdir()
    manifest = _FakeManifest(denominators=_FakeDenominators())

    monkeypatch.setattr(
        cli_dataset,
        "migrate_sol_execbench",
        lambda *args, **kwargs: manifest,
    )
    monkeypatch.setattr(
        cli_dataset,
        "write_migration_manifest",
        lambda manifest_arg, target: Path(target).write_text(manifest_arg.to_json()),
    )

    result = CliRunner().invoke(
        cli,
        [
            "dataset",
            "migrate-sol",
            str(source),
            str(output),
            "--manifest",
            str(explicit_manifest),
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    assert result.output == manifest.to_json()
    assert explicit_manifest.read_text() == manifest.to_json()


def test_dataset_migrate_flashinfer_writes_manifest_and_prints_summary(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    manifest = _FakeManifest(denominators=_FakeDenominators())
    calls: list[dict[str, object]] = []

    def fake_migrate_flashinfer_trace(*args, **kwargs):
        calls.append(
            {
                "source_root": args[0],
                "output_root": args[1],
                "source_revision": kwargs["source_revision"],
            }
        )
        return manifest

    monkeypatch.setattr(
        cli_dataset, "migrate_flashinfer_trace", fake_migrate_flashinfer_trace
    )
    monkeypatch.setattr(
        cli_dataset,
        "write_migration_manifest",
        lambda manifest_arg, target: Path(target).write_text(manifest_arg.to_json()),
    )

    result = CliRunner().invoke(
        cli,
        [
            "dataset",
            "migrate-flashinfer",
            str(source),
            str(output),
            "--source-revision",
            "abc123",
        ],
    )

    assert result.exit_code == 0, result.output
    assert calls == [
        {
            "source_root": source,
            "output_root": output,
            "source_revision": "abc123",
        }
    ]
    assert (output / "migration-manifest.json").read_text() == manifest.to_json()
    assert "Problems:" in result.output
    assert "1 blocker(s)" in result.output


def test_dataset_migrate_flashinfer_prints_json_when_requested(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    manifest = _FakeManifest(denominators=_FakeDenominators())

    monkeypatch.setattr(
        cli_dataset,
        "migrate_flashinfer_trace",
        lambda *args, **kwargs: manifest,
    )
    monkeypatch.setattr(
        cli_dataset,
        "write_migration_manifest",
        lambda manifest_arg, target: Path(target).write_text(manifest_arg.to_json()),
    )

    result = CliRunner().invoke(
        cli,
        [
            "dataset",
            "migrate-flashinfer",
            str(source),
            str(output),
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    assert result.output == manifest.to_json()
```

- [ ] **Step 2: Run dataset tests to verify they fail**

Run:

```bash
uv run pytest tests/sol_execbench/test_cli_dataset_commands.py -q
```

Expected: FAIL during import because `sol_execbench.cli.dataset` does not exist yet.

- [ ] **Step 3: Create `src/sol_execbench/cli/dataset.py`**

Create `src/sol_execbench/cli/dataset.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Dataset migration commands for the SOL-ExecBench CLI."""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console

from ..core.dataset import (
    migrate_flashinfer_trace,
    migrate_sol_execbench,
    write_migration_manifest,
)

console = Console(stderr=True)


@click.group("dataset", context_settings={"help_option_names": ["-h", "--help"]})
def _dataset_cli() -> None:
    """Local dataset utilities."""


@_dataset_cli.command(
    "migrate-sol", context_settings={"help_option_names": ["-h", "--help"]}
)
@click.argument(
    "source_root", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
@click.argument("output_root", type=click.Path(path_type=Path))
@click.option(
    "--category",
    "categories",
    multiple=True,
    help="SOL-ExecBench category to migrate. May be passed more than once.",
)
@click.option("--source-revision", help="Source dataset revision or local commit ref.")
@click.option("--manifest", "manifest_path", type=click.Path(path_type=Path))
@click.option("--json", "json_output", is_flag=True, help="Print manifest JSON")
def _dataset_migrate_sol_cli(
    source_root: Path,
    output_root: Path,
    categories: tuple[str, ...],
    source_revision: str | None,
    manifest_path: Path | None,
    json_output: bool,
) -> None:
    """Migrate downloaded SOL-ExecBench inputs into local benchmark layout."""

    manifest = migrate_sol_execbench(
        source_root,
        output_root,
        categories=categories or None,
        source_revision=source_revision,
    )
    _write_and_report_manifest(
        manifest=manifest,
        target=manifest_path or output_root / "migration-manifest.json",
        json_output=json_output,
    )


@_dataset_cli.command(
    "migrate-flashinfer", context_settings={"help_option_names": ["-h", "--help"]}
)
@click.argument(
    "source_root", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
@click.argument("output_root", type=click.Path(path_type=Path))
@click.option("--source-revision", help="Source dataset revision or local commit ref.")
@click.option("--manifest", "manifest_path", type=click.Path(path_type=Path))
@click.option("--json", "json_output", is_flag=True, help="Print manifest JSON")
def _dataset_migrate_flashinfer_cli(
    source_root: Path,
    output_root: Path,
    source_revision: str | None,
    manifest_path: Path | None,
    json_output: bool,
) -> None:
    """Migrate downloaded FlashInfer Trace inputs into local benchmark layout."""

    manifest = migrate_flashinfer_trace(
        source_root,
        output_root,
        source_revision=source_revision,
    )
    _write_and_report_manifest(
        manifest=manifest,
        target=manifest_path or output_root / "migration-manifest.json",
        json_output=json_output,
    )


def _write_and_report_manifest(
    *,
    manifest,
    target: Path,
    json_output: bool,
) -> None:
    write_migration_manifest(manifest, target)
    if json_output:
        click.echo(manifest.to_json(), nl=False)
    else:
        console.print(f"[green]Wrote migration manifest to {target}[/green]")
        console.print(
            "[bold]Problems:[/bold] "
            f"{manifest.denominators.migrated_problems}/"
            f"{manifest.denominators.discovered_problems} migrated; "
            f"{manifest.denominators.blockers} blocker(s)"
        )
```

- [ ] **Step 4: Update `main.py` dataset dispatch**

In `src/sol_execbench/cli/main.py`, add:

```python
from . import dataset as cli_dataset
```

Remove this import if it becomes unused:

```python
from ..core.dataset import (
    migrate_flashinfer_trace,
    migrate_sol_execbench,
    write_migration_manifest,
)
```

Delete `_dataset_cli`, `_dataset_migrate_sol_cli`, and `_dataset_migrate_flashinfer_cli` from `main.py`.

Replace:

```python
return _dataset_cli.main(
```

with:

```python
return cli_dataset._dataset_cli.main(
```

- [ ] **Step 5: Run dataset and full boundary tests**

Run:

```bash
uv run pytest tests/sol_execbench/test_cli_dataset_commands.py tests/sol_execbench/test_cli_module_boundaries.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/sol_execbench/cli/dataset.py src/sol_execbench/cli/main.py tests/sol_execbench/test_cli_dataset_commands.py tests/sol_execbench/test_cli_module_boundaries.py
git commit -s -m "#0 - Extract CLI dataset commands"
```

---

### Task 5: Run command-surface regression checks

**Files:**
- No source edits expected unless Ruff reports import-order or lint issues.

- [ ] **Step 1: Run focused CLI command tests**

Run:

```bash
uv run pytest tests/sol_execbench/test_cli_metadata_commands.py tests/sol_execbench/test_cli_baseline_commands.py tests/sol_execbench/test_cli_dataset_commands.py tests/sol_execbench/test_cli_module_boundaries.py -q
```

Expected: PASS.

- [ ] **Step 2: Run existing CLI diagnostics/evaluator tests**

Run:

```bash
uv run pytest tests/sol_execbench/test_cli_diagnostics.py tests/sol_execbench/test_cli_environment.py tests/sol_execbench/test_cli_reporting.py tests/sol_execbench/test_cli_evaluation_timeout.py -q
```

Expected: PASS.

- [ ] **Step 3: Run contract and docs guard tests touched by metadata commands**

Run:

```bash
uv run pytest tests/sol_execbench/test_contract.py tests/sol_execbench/test_public_contract_guardrails.py -q
```

Expected: PASS.

- [ ] **Step 4: Run Ruff on changed command modules and tests**

Run:

```bash
uv run --with ruff ruff check src/sol_execbench/cli/main.py src/sol_execbench/cli/metadata.py src/sol_execbench/cli/baseline.py src/sol_execbench/cli/dataset.py tests/sol_execbench/test_cli_metadata_commands.py tests/sol_execbench/test_cli_baseline_commands.py tests/sol_execbench/test_cli_dataset_commands.py tests/sol_execbench/test_cli_diagnostics.py tests/sol_execbench/test_cli_module_boundaries.py
```

Expected: PASS.

- [ ] **Step 5: Commit Ruff-only cleanup if needed**

Only run this if Step 4 required import-order or lint cleanup:

```bash
git add src/sol_execbench/cli/main.py src/sol_execbench/cli/metadata.py src/sol_execbench/cli/baseline.py src/sol_execbench/cli/dataset.py tests/sol_execbench/test_cli_metadata_commands.py tests/sol_execbench/test_cli_baseline_commands.py tests/sol_execbench/test_cli_dataset_commands.py tests/sol_execbench/test_cli_diagnostics.py tests/sol_execbench/test_cli_module_boundaries.py
git commit -s -m "#0 - Verify CLI command surface refactor"
```

---

## Expected End State

- `src/sol_execbench/cli/main.py` owns evaluator command orchestration, root dispatch, and top-level `cli`.
- `src/sol_execbench/cli/metadata.py` owns GPU-free metadata commands.
- `src/sol_execbench/cli/baseline.py` owns measured baseline export commands.
- `src/sol_execbench/cli/dataset.py` owns local dataset migration commands.
- Existing command syntax remains unchanged:
  - `sol-execbench contract --json`
  - `sol-execbench doctor --json`
  - `sol-execbench toolchain --json`
  - `sol-execbench baseline export ...`
  - `sol-execbench dataset migrate-sol ...`
  - `sol-execbench dataset migrate-flashinfer ...`
- `tests/sol_execbench/test_cli_module_boundaries.py` guards that these commands do not drift back into `main.py`.

---

## Self-Review

**Spec coverage:** This plan covers the requested next refactor from recent-commit analysis: split non-evaluation CLI command surfaces before touching `_evaluate_cli`. It explicitly excludes GSD and leaves the evaluator path untouched.

**Placeholder scan:** No placeholder items are left. Each task has exact files, code snippets, commands, expected outcomes, and commit messages.

**Type consistency:** Command object names are stable across tasks: `_contract_cli`, `_doctor_cli`, `_toolchain_cli`, `_baseline_cli`, `_baseline_export_cli`, `_dataset_cli`, `_dataset_migrate_sol_cli`, and `_dataset_migrate_flashinfer_cli`. Module aliases used from `main.py` are consistently `cli_metadata`, `cli_baseline`, and `cli_dataset`.
