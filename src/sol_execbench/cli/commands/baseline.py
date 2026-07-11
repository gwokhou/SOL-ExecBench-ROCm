# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Baseline-related CLI commands."""

from __future__ import annotations

import json
from pathlib import Path

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
    load_release_baseline_bundle,
    sha256_file,
    verify_release_baseline_rerun,
    write_release_baseline_outputs,
    write_release_baseline_verification,
)


console = Console(stderr=True)


@click.group("baseline", context_settings={"help_option_names": ["-h", "--help"]})
def _baseline_cli() -> None:
    """Measured baseline export utilities."""


def _load_json(path: Path, description: str) -> object:
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
        for index, raw in enumerate(payload):
            if not isinstance(raw, dict):
                raise ValueError(f"authority workload {index} must be an object")
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


@_baseline_cli.command("release-build")
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
    latency_tolerance_rel: float,
) -> None:
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


@_baseline_cli.command("release-verify")
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
) -> None:
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


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
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
    "--format",
    "output_format",
    type=click.Choice(["text", "json"]),
    default="text",
    show_default=True,
    help="Output format.",
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
def cli(
    candidate_file: Path,
    baseline_files: tuple[Path, ...],
    output_format: str,
    output_file: Path | None,
    win_pct: float,
    parity_pct: float,
    amd_native_claim: bool,
) -> None:
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

    if output_format == "json":
        rendered = json.dumps(comparison_to_json(comparison), indent=2)
    else:
        rendered = format_baseline_comparison(comparison)

    if output_file is not None:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(rendered + "\n")
    else:
        click.echo(rendered)


if __name__ == "__main__":
    cli()
