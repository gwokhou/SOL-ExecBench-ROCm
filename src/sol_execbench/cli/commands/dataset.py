# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""AKA-derived problem-corpus commands.

Materialize and audit the problem set derived from AMD AgentKernelArena (AKA).
The benchmark problems are authored artifacts committed under
``problems/AMD_AKA/<suite>/<name>/``. ``materialize`` selects workloads for an
observed GPU and ``audit`` verifies the target-specific result.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import click
from rich.console import Console

from sol_execbench.cli.protocol import (
    CliExitCode,
    CliFailure,
    CliResult,
    artifact,
)
from sol_execbench.core.dataset.aka_compatibility import (
    DEFAULT_PROBE_TIMEOUT_SECONDS,
    SUPPORTED_AKA_GFX_TARGETS,
    AKAProbeInfrastructureError,
    materialization_target,
)
from sol_execbench.core.dataset.aka_corpus import (
    AKA_REVISION,
    AKACorpusManifest,
)
from sol_execbench.core.dataset.corpus import (
    SELECTION_MANIFEST_FILENAME,
    load_target_descriptor,
    select_corpus,
    validate_corpus,
)
from sol_execbench.core.dataset.corpus_models import CorpusProfile
from sol_execbench.core.platform.runtime import detect_rocm_device

console = Console(stderr=True)
DEFAULT_MANIFEST = Path("problems/AMD_AKA/manifest.yaml")
DEFAULT_OUTPUT_ROOT = Path("problems/local/AMD_AKA")
DEFAULT_AKA_ROOT = Path("data/AgentKernelArena")
DEFAULT_FETCH_SCRIPT = Path("scripts/fetch_aka_source.sh")
DEFAULT_CORPUS_MANIFEST = Path(
    "problems/LLM_CORE/releases/LLM_CORE_V1/manifest.yaml",
)
DEFAULT_TARGET_ROOT = Path("problems/LLM_CORE/targets")
TARGET_TEMPLATES = ("gfx1200", "gfx942")


@click.group(
    "dataset",
    context_settings={"help_option_names": ["-h", "--help"]},
)
def dataset_cli() -> None:
    """Validate and select problem corpora; preserve AKA compatibility."""


@dataset_cli.group("corpus")
def corpus_cli() -> None:
    """Validate or statically select a hardware-independent corpus."""


@corpus_cli.command("validate")
@click.option(
    "--manifest",
    "manifest_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=DEFAULT_CORPUS_MANIFEST,
    show_default=True,
)
def validate_corpus_cli(manifest_path: Path) -> CliResult:
    """Validate the manifest, artifacts, provenance, and coverage floors."""
    try:
        report = validate_corpus(manifest_path)
    except (OSError, ValueError) as exc:
        raise CliFailure(str(exc), code="invalid_corpus_manifest") from exc
    console.print(
        f"[green]Valid {report['release_id']}: {report['definitions']} definitions, "
        f"{report['workloads']} workloads[/green]",
    )
    return CliResult(data={"manifest": str(manifest_path), **report})


@corpus_cli.command("select")
@click.option(
    "--manifest",
    "manifest_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=DEFAULT_CORPUS_MANIFEST,
    show_default=True,
)
@click.option(
    "--target-template",
    type=click.Choice(TARGET_TEMPLATES),
    help="Bundled declared target template.",
)
@click.option(
    "--target-descriptor",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="External static target descriptor in JSON or YAML.",
)
@click.option(
    "--memory-budget",
    type=click.IntRange(min=1),
    help="Usable bytes; required when the descriptor does not declare a budget.",
)
@click.option(
    "--profile",
    "profile_values",
    type=click.Choice(tuple(profile.value for profile in CorpusProfile)),
    multiple=True,
    default=(CorpusProfile.CORE.value,),
    show_default=True,
)
@click.option(
    "--output",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
)
@click.option(
    "--require-complete-profile",
    is_flag=True,
    help="Fail when the selected target falls below a requested profile floor.",
)
def select_corpus_cli(
    manifest_path: Path,
    target_template: str | None,
    target_descriptor: Path | None,
    memory_budget: int | None,
    profile_values: tuple[str, ...],
    output: Path,
    require_complete_profile: bool,
) -> CliResult:
    """Create a deterministic target view without probing a GPU."""
    if (target_template is None) == (target_descriptor is None):
        raise CliFailure(
            "provide exactly one of --target-template or --target-descriptor",
            code="invalid_static_target",
        )
    target_path = (
        target_descriptor or DEFAULT_TARGET_ROOT / f"{target_template}.yaml"
    )
    try:
        target = load_target_descriptor(target_path)
        if memory_budget is not None:
            target = target.model_copy(
                update={"memory_budget_bytes": memory_budget},
            )
        if target.memory_budget_bytes is None:
            raise ValueError(
                "--memory-budget is required for this target descriptor"
            )
        result = select_corpus(
            manifest_path,
            output,
            target=target,
            profiles=tuple(CorpusProfile(value) for value in profile_values),
            require_complete_profile=require_complete_profile,
        )
    except FileExistsError as exc:
        raise CliFailure(
            str(exc), code="corpus_selection_output_exists"
        ) from exc
    except (OSError, ValueError) as exc:
        raise CliFailure(str(exc), code="corpus_selection_invalid") from exc
    record = result / SELECTION_MANIFEST_FILENAME
    console.print(f"[green]Static corpus selection written to {result}[/green]")
    return CliResult(
        data={
            "output": str(result),
            "target_id": target.target_id,
            "profiles": list(profile_values),
            "qualification_status": target.qualification_status.value,
        },
        artifacts=(artifact(record, "yaml_file"),),
    )


@dataset_cli.command("materialize")
@click.option(
    "--manifest",
    "manifest_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=DEFAULT_MANIFEST,
    show_default=True,
)
@click.option(
    "--aka-root",
    type=click.Path(file_okay=False, path_type=Path),
    default=DEFAULT_AKA_ROOT,
    show_default=True,
    help="Local clone of AgentKernelArena; fetched when absent or stale.",
)
@click.option(
    "--output",
    type=click.Path(file_okay=False, path_type=Path),
    help="Output tree; defaults to problems/local/AMD_AKA/<detected-gfx>.",
)
@click.option(
    "--device",
    default="cuda:0",
    show_default=True,
    help="ROCm PyTorch device used for target detection and live probes.",
)
@click.option(
    "--target-arch",
    type=click.Choice(SUPPORTED_AKA_GFX_TARGETS),
    help="Expected exact gfx target; fail if it differs from the detected device.",
)
@click.option(
    "--probe-timeout",
    "probe_timeout_seconds",
    type=click.FloatRange(min=1.0),
    default=DEFAULT_PROBE_TIMEOUT_SECONDS,
    show_default=True,
    help="Per-workload live-probe timeout in seconds.",
)
@click.option(
    "--skip-aka-fetch",
    is_flag=True,
    default=False,
    help="Do not fetch/verify the AKA clone; only mirror authored problems.",
)
def materialize_cli(
    manifest_path: Path,
    aka_root: Path,
    output: Path | None,
    device: str,
    target_arch: str | None,
    probe_timeout_seconds: float,
    skip_aka_fetch: bool,
) -> CliResult:
    """Select executable AKA workloads for one exact AMD GPU target."""
    manifest = _load_manifest(manifest_path)
    try:
        device_info = detect_rocm_device(device)
        target = materialization_target(device_info)
    except (RuntimeError, ValueError) as exc:
        raise CliFailure(
            str(exc),
            code="aka_target_unavailable",
            exit_code=CliExitCode.UNAVAILABLE,
            hint="Use a ROCm device whose exact target is gfx942, gfx1150, or gfx1200.",
        ) from exc
    if target_arch is not None and device_info.gfx_target != target_arch:
        raise CliFailure(
            f"detected {device_info.gfx_target} on {device}, expected {target_arch}",
            code="aka_target_mismatch",
            hint="Remove --target-arch or select the matching GPU with --device.",
        )
    if not skip_aka_fetch:
        _ensure_aka_clone(aka_root)
    output = output or DEFAULT_OUTPUT_ROOT / device_info.gfx_target
    try:
        result_path = manifest.materialize(
            output,
            target=target,
            probe_timeout_seconds=probe_timeout_seconds,
        )
    except AKAProbeInfrastructureError as exc:
        raise CliFailure(
            str(exc),
            code="aka_probe_infrastructure_error",
            exit_code=CliExitCode.UNAVAILABLE,
            hint="Check ROCm visibility and retry the reported workload probe.",
        ) from exc
    except FileExistsError as exc:
        raise CliFailure(
            str(exc),
            code="aka_materialization_output_exists",
            hint="Choose a new --output path or remove the old tree after auditing it.",
        ) from exc
    except ValueError as exc:
        raise CliFailure(str(exc), code="aka_materialization_invalid") from exc
    report = manifest.audit(result_path)
    console.print(
        f"[green]Materialized {report['problems']} problems / "
        f"{report['workloads']} workloads for {report['gfx_target']} in {result_path} "
        f"({report['excluded_workloads']} workloads excluded)"
        f"[/green]",
    )
    record = result_path / "materialization-manifest.yaml"
    return CliResult(
        data={"output": str(result_path), **report},
        artifacts=(artifact(record, "yaml_file"),),
    )


@dataset_cli.command("audit")
@click.option(
    "--manifest",
    "manifest_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=DEFAULT_MANIFEST,
    show_default=True,
)
@click.option(
    "--aka-root",
    type=click.Path(file_okay=False, path_type=Path),
    default=DEFAULT_AKA_ROOT,
    show_default=True,
    help="Local AKA clone; when present, verify it is pinned and matches every "
    "entry's per-task checksums (problems-bound-to-commit binding).",
)
@click.argument(
    "problem_root",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
)
def audit_cli(
    manifest_path: Path,
    aka_root: Path,
    problem_root: Path,
) -> CliResult:
    """Fail closed if local problems differ from the pinned AKA selection."""
    manifest = _load_manifest(manifest_path)
    try:
        report = manifest.audit(problem_root)
    except (OSError, ValueError) as exc:
        raise CliFailure(str(exc), code="aka_audit_failed") from exc
    if aka_root.is_dir():
        report["aka_provenance"] = manifest.audit_aka_provenance(aka_root)
        console.print(
            f"[green]AKA provenance bound to {report['aka_provenance']['revision'][:12]} "
            f"({report['aka_provenance']['checksums_verified']} checksums verified)[/green]",
        )
    console.print(
        f"[green]Valid AKA corpus: {report['problems']} problems, "
        f"{report['scored']} scored[/green]",
    )
    return CliResult(data={"problem_root": str(problem_root), **report})


def _load_manifest(path: Path) -> AKACorpusManifest:
    try:
        return AKACorpusManifest.load(path)
    except (OSError, ValueError) as exc:
        raise CliFailure(str(exc), code="invalid_aka_manifest") from exc


def _ensure_aka_clone(aka_root: Path) -> None:
    """Ensure the AKA clone is present at the pinned revision (best-effort)."""
    head_file = aka_root / ".aka-head"
    if head_file.is_file():
        try:
            if head_file.read_text().strip() == AKA_REVISION:
                return
        except OSError:
            pass
    if not DEFAULT_FETCH_SCRIPT.is_file():
        console.print(
            f"[yellow]AKA fetch script missing at {DEFAULT_FETCH_SCRIPT}; "
            f"continuing without verifying the AKA clone.[/yellow]",
        )
        return
    try:
        subprocess.run(["bash", str(DEFAULT_FETCH_SCRIPT)], check=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        console.print(
            f"[yellow]Could not fetch/verify AKA clone ({exc}); "
            f"continuing with authored problems only.[/yellow]",
        )


__all__ = ["dataset_cli"]
