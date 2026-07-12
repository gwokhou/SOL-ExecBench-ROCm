# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Baseline-related CLI commands."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import click
from rich.console import Console

from ...core.evidence.baseline_export import export_hip_baseline_registry
from ...core.evidence.baseline import (
    compare_trace_baselines,
    comparison_to_json,
    format_baseline_comparison,
    load_trace_jsonl,
)
from ...core.scoring.release_baseline import (
    AuthorityInput,
    ReleaseProvenance,
    build_release_baseline_bundle,
    load_evidence_publication_manifest,
    load_release_baseline_bundle,
    sha256_file,
    verify_release_baseline_rerun,
    write_release_baseline_outputs,
    write_release_baseline_verification,
)
from ...core.scoring.authority_slice import (
    AuthoritySliceSelectionPolicy,
    build_authority_slice_manifest,
    write_authority_slice_manifest,
)
from ...core.scoring.representative_suite import (
    build_representative_suite_manifest,
)
from ...core.scoring.baseline_selection import (
    baseline_candidates_from_dict,
    build_baseline_selection_manifest,
)
from ...core.scoring.amd_bound_sanity.models import AmdBoundSanityReport
from ..protocol import CliResult, artifact, output_format


console = Console(stderr=True)


@click.group("baseline", context_settings={"help_option_names": ["-h", "--help"]})
def baseline_cli() -> None:
    """Measured baseline export utilities."""


def _load_json(path: Path, description: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise click.ClickException(f"invalid {description} JSON: {exc}") from exc


def _suite_workloads_from_json(path: Path) -> list[dict[str, str]]:
    payload = _load_json(path, "suite manifest")
    if isinstance(payload, dict):
        payload = payload.get("workloads")
    if not isinstance(payload, list):
        raise click.ClickException(
            "suite manifest must be a JSON list or object with a workloads list"
        )
    return payload


def _authority_from_json(path: Path | None) -> dict[tuple[str, str], AuthorityInput]:
    if path is None:
        return {}
    payload = _load_json(path, "authority")
    if isinstance(payload, dict):
        payload = payload.get("workloads")
    if not isinstance(payload, list):
        raise click.ClickException(
            "authority JSON must be a list or object with a workloads list"
        )

    authority: dict[tuple[str, str], AuthorityInput] = {}
    try:
        for index, item in enumerate(payload):
            if not isinstance(item, dict):
                raise ValueError(f"authority workload {index} must be an object")
            raw: Any = item
            definition = raw["definition"]
            workload_uuid = raw["workload_uuid"]
            if not isinstance(definition, str) or not isinstance(workload_uuid, str):
                raise ValueError(
                    f"authority workload {index} requires string definition and workload_uuid"
                )
            authority[(definition, workload_uuid)] = AuthorityInput(
                official_blockers=tuple(raw.get("official_blockers", ())),
                bound_ref=raw.get("bound_ref"),
                bound_sha256=raw.get("bound_sha256"),
                hardware_model_ref=raw.get("hardware_model_ref"),
                hardware_model_sha256=raw.get("hardware_model_sha256"),
            )
    except (KeyError, TypeError, ValueError) as exc:
        raise click.ClickException(f"invalid authority JSON: {exc}") from exc
    return authority


@baseline_cli.group("authority")
def authority_cli() -> None:
    """Manage frozen baseline authority inputs."""


@baseline_cli.group("suite")
def suite_cli() -> None:
    """Freeze named score-suite denominators."""


@baseline_cli.group("selection")
def selection_cli() -> None:
    """Freeze deterministic winner selection for a baseline portfolio."""


@suite_cli.command("freeze-representative-gfx1200")
@click.option(
    "--benchmark-root",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option(
    "--output", required=True, type=click.Path(dir_okay=False, path_type=Path)
)
def _freeze_representative_gfx1200_cli(benchmark_root: Path, output: Path) -> CliResult:
    """Write the 87-workload gfx1200 representative suite manifest."""
    try:
        manifest = build_representative_suite_manifest(benchmark_root)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    return CliResult(
        data={"scope": manifest["scope"], "workloads": len(manifest["workloads"])},
        artifacts=(artifact(output, "json_file"),),
    )


@selection_cli.command("build")
@click.option(
    "--suite-manifest",
    "suite_manifest_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--candidates",
    "candidates_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option("--scope", required=True)
@click.option(
    "--output", required=True, type=click.Path(dir_okay=False, path_type=Path)
)
def _selection_build_cli(
    suite_manifest_path: Path,
    candidates_path: Path,
    scope: str,
    output: Path,
) -> CliResult:
    """Select exactly one baseline candidate for every frozen workload."""
    try:
        workloads = _suite_workloads_from_json(suite_manifest_path)
        required_keys = []
        for index, workload in enumerate(workloads):
            if not isinstance(workload, dict):
                raise ValueError(f"suite workload {index} must be an object")
            definition = workload.get("definition")
            workload_uuid = workload.get("workload_uuid")
            if not isinstance(definition, str) or not isinstance(workload_uuid, str):
                raise ValueError(
                    f"suite workload {index} requires definition and workload_uuid"
                )
            required_keys.append((definition, workload_uuid))
        manifest = build_baseline_selection_manifest(
            scope=scope,
            candidates=baseline_candidates_from_dict(
                _load_json(candidates_path, "baseline candidates")
            ),
            required_workload_keys=required_keys,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    return CliResult(
        data={"selections": len(manifest.selections), "scope": manifest.scope},
        artifacts=(artifact(output, "json_file"),),
    )


@authority_cli.command("freeze")
@click.option(
    "--suite-manifest",
    "suite_manifest_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--sanity-report",
    "sanity_report_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--output",
    "output_path",
    required=True,
    type=click.Path(dir_okay=False, path_type=Path),
)
@click.option(
    "--selection-policy-version",
    default="amd-authority-scored-evidence-v1",
    show_default=True,
)
def _authority_freeze_cli(
    suite_manifest_path: Path,
    sanity_report_path: Path,
    output_path: Path,
    selection_policy_version: str,
) -> CliResult:
    """Freeze a score-independent authoritative workload subset."""
    try:
        sanity_payload = _load_json(sanity_report_path, "AMD bound sanity report")
        if not isinstance(sanity_payload, dict):
            raise ValueError("AMD bound sanity report must be an object")
        manifest = build_authority_slice_manifest(
            suite_workloads=_suite_workloads_from_json(suite_manifest_path),
            source_suite_manifest_sha256=sha256_file(suite_manifest_path),
            sanity_report=AmdBoundSanityReport.model_validate(sanity_payload),
            selection_policy=AuthoritySliceSelectionPolicy(
                version=selection_policy_version
            ),
        )
        write_authority_slice_manifest(manifest, output_path)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    console.print(
        f"[green]Wrote frozen authority slice to {output_path}[/green] "
        f"({len(manifest.workloads)} selected, {len(manifest.excluded)} excluded)"
    )
    return CliResult(
        data={"selected": len(manifest.workloads), "excluded": len(manifest.excluded)},
        artifacts=(artifact(output_path, "json_file"),),
    )


@baseline_cli.group("release")
def release_cli() -> None:
    """Build and verify immutable release baseline evidence."""


@release_cli.command("build")
@click.option(
    "--suite-manifest",
    "suite_manifest_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--trace",
    "trace_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option("--release", required=True)
@click.option(
    "--baseline-output",
    "baseline_output_path",
    required=True,
    type=click.Path(dir_okay=False, path_type=Path),
)
@click.option(
    "--bundle-output",
    "bundle_output_path",
    required=True,
    type=click.Path(dir_okay=False, path_type=Path),
)
@click.option(
    "--authority-json",
    "authority_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option("--solution", required=True)
@click.option("--solution-sha256", required=True)
@click.option("--environment-fingerprint", required=True)
@click.option("--clock-policy", required=True)
@click.option("--timing-policy", required=True)
@click.option("--compiler-build-id", required=True)
@click.option(
    "--scope",
    required=True,
    help="Human-readable authority scope, for example authority-slice:gfx1200:gemm:25-workloads.",
)
@click.option(
    "--latency-tolerance-rel",
    required=True,
    type=click.FloatRange(min=0.0, min_open=True),
)
def _release_build_cli(
    suite_manifest_path: Path,
    trace_path: Path,
    release: str,
    baseline_output_path: Path,
    bundle_output_path: Path,
    authority_path: Path | None,
    solution: str,
    solution_sha256: str,
    environment_fingerprint: str,
    clock_policy: str,
    timing_policy: str,
    compiler_build_id: str,
    scope: str,
    latency_tolerance_rel: float,
) -> CliResult:
    """Build compact and complete release-baseline evidence from one trace."""

    suite_workloads = _suite_workloads_from_json(suite_manifest_path)
    provenance = ReleaseProvenance(
        solution=solution,
        solution_sha256=solution_sha256,
        environment_fingerprint=environment_fingerprint,
        clock_policy=clock_policy,
        compiler_build_id=compiler_build_id,
        timing_policy=timing_policy,
        suite_manifest_sha256=sha256_file(suite_manifest_path),
    )
    try:
        baseline, bundle = build_release_baseline_bundle(
            suite_workloads=suite_workloads,
            trace_path=trace_path,
            release=release,
            provenance=provenance,
            authority_by_key=_authority_from_json(authority_path),
            latency_tolerance_rel=latency_tolerance_rel,
            suite_manifest_ref=str(suite_manifest_path),
            suite_manifest_sha256=provenance.suite_manifest_sha256,
            scope=scope,
        )
        written_bundle = write_release_baseline_outputs(
            baseline=baseline,
            bundle=bundle,
            baseline_path=baseline_output_path,
            bundle_path=bundle_output_path,
        )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    console.print(f"[green]Wrote scoring baseline to {baseline_output_path}[/green]")
    console.print(
        f"[green]Wrote release baseline bundle to {bundle_output_path}[/green]"
    )
    console.print(f"Release {written_bundle.release}: {written_bundle.summary}")
    return CliResult(
        data={"release": written_bundle.release, "summary": written_bundle.summary},
        artifacts=(
            artifact(baseline_output_path, "json_file"),
            artifact(bundle_output_path, "json_file"),
        ),
    )


@release_cli.command("verify")
@click.option(
    "--bundle",
    "bundle_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--rerun-trace",
    "rerun_trace_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--output",
    "output_path",
    required=True,
    type=click.Path(dir_okay=False, path_type=Path),
)
@click.option("--solution-sha256", required=True)
@click.option("--environment-fingerprint", required=True)
@click.option("--clock-policy", required=True)
@click.option("--timing-policy", required=True)
@click.option("--compiler-build-id", required=True)
@click.option("--suite-manifest-sha256", required=True)
def _release_verify_cli(
    bundle_path: Path,
    rerun_trace_path: Path,
    output_path: Path,
    solution_sha256: str,
    environment_fingerprint: str,
    clock_policy: str,
    timing_policy: str,
    compiler_build_id: str,
    suite_manifest_sha256: str,
) -> CliResult:
    """Verify an immutable release baseline bundle against a rerun trace."""

    try:
        bundle = load_release_baseline_bundle(bundle_path)
        report = verify_release_baseline_rerun(
            bundle=bundle,
            bundle_path=bundle_path,
            rerun_trace_path=rerun_trace_path,
            rerun_provenance=ReleaseProvenance(
                solution=bundle.provenance.solution,
                solution_sha256=solution_sha256,
                environment_fingerprint=environment_fingerprint,
                clock_policy=clock_policy,
                compiler_build_id=compiler_build_id,
                timing_policy=timing_policy,
                suite_manifest_sha256=suite_manifest_sha256,
            ),
        )
        write_release_baseline_verification(report, output_path)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    console.print(
        f"[green]Wrote release baseline verification to {output_path}[/green]"
    )
    console.print(f"Release {report.release}: {report.summary}")
    return CliResult(
        data={"release": report.release, "summary": report.summary},
        artifacts=(artifact(output_path, "json_file"),),
        exit_code=(
            1
            if isinstance(report.summary, dict)
            and report.summary.get("passed") != report.summary.get("total")
            else 0
        ),
    )


@baseline_cli.group("publication")
def publication_cli() -> None:
    """Verify published evidence artifacts."""


@publication_cli.command("verify")
@click.option(
    "--manifest",
    "manifest_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Git-tracked evidence_publication_manifest.v1 JSON.",
)
@click.option(
    "--artifact-root",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Directory containing the downloaded release artifacts.",
)
def _publication_verify_cli(manifest_path: Path, artifact_root: Path) -> CliResult:
    """Verify a downloaded evidence bundle against its Git-tracked manifest."""
    try:
        manifest = load_evidence_publication_manifest(manifest_path)
        manifest.verify_artifact_root(artifact_root)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    console.print(
        f"[green]Verified published evidence for {manifest.release} ({manifest.scope})[/green]"
    )
    return CliResult(data={"release": manifest.release, "scope": manifest.scope})


@baseline_cli.command(
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
def _baseline_export_cli(
    trace_path: Path,
    output_path: Path,
    target_id: str,
    sol_version: str,
    timing_policy: str,
) -> CliResult:
    """Export a HIP measured baseline registry from a SOL trace JSONL file."""

    registry = export_hip_baseline_registry(
        trace_path=trace_path,
        output_path=output_path,
        target_id=target_id,
        sol_version=sol_version,
        timing_policy=timing_policy,
    )
    if output_format() == "text":
        console.print(
            f"[green]Wrote measured baseline registry to {output_path}[/green]"
        )
    return CliResult(
        data={"registry": registry},
        artifacts=(artifact(output_path, "json_file"),),
    )


@baseline_cli.command(
    "compare", context_settings={"help_option_names": ["-h", "--help"]}
)
@click.option(
    "--candidate",
    "candidate_file",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Candidate trace JSONL file.",
)
@click.option(
    "--baseline",
    "baseline_files",
    required=True,
    multiple=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Baseline trace JSONL file. May be repeated.",
)
@click.option(
    "--output",
    "output_file",
    type=click.Path(path_type=Path),
    help="Optional output path. Defaults to stdout.",
)
@click.option(
    "--win-pct",
    default=2.0,
    show_default=True,
    type=float,
    help="Candidate must beat baseline by at least this percentage to be a WIN.",
)
@click.option(
    "--parity-pct",
    default=5.0,
    show_default=True,
    type=float,
    help="Candidate within this percentage of baseline is PARITY.",
)
@click.option(
    "--amd-native-claim",
    is_flag=True,
    help="Label output as an AMD-native claim and emit guardrail warnings.",
)
def _baseline_compare_cli(
    candidate_file: Path,
    baseline_files: tuple[Path, ...],
    output_file: Path | None,
    win_pct: float,
    parity_pct: float,
    amd_native_claim: bool,
) -> CliResult:
    """Compare candidate trace JSONL against one or more baseline trace files."""
    candidate_traces = load_trace_jsonl(candidate_file)
    baseline_traces = []
    for baseline_file in baseline_files:
        baseline_traces.extend(load_trace_jsonl(baseline_file))

    comparison = compare_trace_baselines(
        candidate_traces,
        baseline_traces,
        win_threshold_pct=win_pct,
        parity_threshold_pct=parity_pct,
        amd_native_claim=amd_native_claim,
    )

    payload = comparison_to_json(comparison)
    if output_format() == "json":
        rendered = json.dumps(payload, indent=2)
    else:
        rendered = format_baseline_comparison(comparison)

    if output_file is not None:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(rendered + "\n")
    else:
        click.echo(rendered)
    artifacts = (artifact(output_file, "json_file"),) if output_file else ()
    return CliResult(data=payload, artifacts=artifacts)


_baseline_cli = baseline_cli
