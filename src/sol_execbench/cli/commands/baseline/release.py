"""Release-baseline Click adapters for the baseline command group."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import click
from rich.console import Console

from sol_execbench.cli.commands.baseline.inputs import (
    authority_from_json,
    suite_workloads_from_json,
)
from sol_execbench.cli.protocol import CliResult, artifact
from sol_execbench.core.scoring.release_baseline import (
    ReleaseProvenance,
    build_release_baseline_bundle,
    load_release_baseline_bundle,
    sha256_file,
    verify_release_baseline_rerun,
    write_release_baseline_outputs,
    write_release_baseline_verification,
)

console = Console(stderr=True)


@dataclass(frozen=True, slots=True)
class ReleaseBuildRequest:
    """Complete input contract for one release-baseline build."""

    suite_manifest_path: Path
    trace_path: Path
    release: str
    baseline_output_path: Path
    bundle_output_path: Path
    authority_path: Path | None
    solution: str
    solution_sha256: str
    environment_fingerprint: str
    clock_policy: str
    timing_policy: str
    compiler_build_id: str
    scope: str
    latency_tolerance_rel: float


@dataclass(frozen=True, slots=True)
class ReleaseVerifyRequest:
    """Complete input contract for one release-baseline rerun verification."""

    bundle_path: Path
    rerun_trace_path: Path
    output_path: Path
    solution_sha256: str
    environment_fingerprint: str
    clock_policy: str
    timing_policy: str
    compiler_build_id: str
    suite_manifest_sha256: str


def register_release_commands(baseline_cli: click.Group) -> None:
    """Attach immutable release evidence commands to the baseline group."""

    @baseline_cli.group("release")
    def release_cli() -> None:
        """Build and verify immutable release baseline evidence."""

    release_cli.add_command(build_cli)
    release_cli.add_command(verify_cli)


@click.command("build")
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
def build_cli(
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
    return _build_release(
        ReleaseBuildRequest(
            suite_manifest_path=suite_manifest_path,
            trace_path=trace_path,
            release=release,
            baseline_output_path=baseline_output_path,
            bundle_output_path=bundle_output_path,
            authority_path=authority_path,
            solution=solution,
            solution_sha256=solution_sha256,
            environment_fingerprint=environment_fingerprint,
            clock_policy=clock_policy,
            timing_policy=timing_policy,
            compiler_build_id=compiler_build_id,
            scope=scope,
            latency_tolerance_rel=latency_tolerance_rel,
        )
    )


def _build_release(request: ReleaseBuildRequest) -> CliResult:
    provenance = ReleaseProvenance(
        solution=request.solution,
        solution_sha256=request.solution_sha256,
        environment_fingerprint=request.environment_fingerprint,
        clock_policy=request.clock_policy,
        compiler_build_id=request.compiler_build_id,
        timing_policy=request.timing_policy,
        suite_manifest_sha256=sha256_file(request.suite_manifest_path),
    )
    try:
        baseline, bundle = build_release_baseline_bundle(
            suite_workloads=suite_workloads_from_json(request.suite_manifest_path),
            trace_path=request.trace_path,
            release=request.release,
            provenance=provenance,
            authority_by_key=authority_from_json(request.authority_path),
            latency_tolerance_rel=request.latency_tolerance_rel,
            suite_manifest_ref=str(request.suite_manifest_path),
            suite_manifest_sha256=provenance.suite_manifest_sha256,
            scope=request.scope,
        )
        written_bundle = write_release_baseline_outputs(
            baseline=baseline,
            bundle=bundle,
            baseline_path=request.baseline_output_path,
            bundle_path=request.bundle_output_path,
        )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    console.print(
        f"[green]Wrote scoring baseline to {request.baseline_output_path}[/green]"
    )
    console.print(
        f"[green]Wrote release baseline bundle to {request.bundle_output_path}[/green]"
    )
    console.print(f"Release {written_bundle.release}: {written_bundle.summary}")
    return CliResult(
        data={"release": written_bundle.release, "summary": written_bundle.summary},
        artifacts=(
            artifact(request.baseline_output_path, "json_file"),
            artifact(request.bundle_output_path, "json_file"),
        ),
    )


@click.command("verify")
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
def verify_cli(
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
    return _verify_release(
        ReleaseVerifyRequest(
            bundle_path=bundle_path,
            rerun_trace_path=rerun_trace_path,
            output_path=output_path,
            solution_sha256=solution_sha256,
            environment_fingerprint=environment_fingerprint,
            clock_policy=clock_policy,
            timing_policy=timing_policy,
            compiler_build_id=compiler_build_id,
            suite_manifest_sha256=suite_manifest_sha256,
        )
    )


def _verify_release(request: ReleaseVerifyRequest) -> CliResult:
    try:
        bundle = load_release_baseline_bundle(request.bundle_path)
        report = verify_release_baseline_rerun(
            bundle=bundle,
            bundle_path=request.bundle_path,
            rerun_trace_path=request.rerun_trace_path,
            rerun_provenance=ReleaseProvenance(
                solution=bundle.provenance.solution,
                solution_sha256=request.solution_sha256,
                environment_fingerprint=request.environment_fingerprint,
                clock_policy=request.clock_policy,
                compiler_build_id=request.compiler_build_id,
                timing_policy=request.timing_policy,
                suite_manifest_sha256=request.suite_manifest_sha256,
            ),
        )
        write_release_baseline_verification(report, request.output_path)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    console.print(
        f"[green]Wrote release baseline verification to {request.output_path}[/green]"
    )
    console.print(f"Release {report.release}: {report.summary}")
    return CliResult(
        data={"release": report.release, "summary": report.summary},
        artifacts=(artifact(request.output_path, "json_file"),),
        exit_code=(1 if report.summary["passed"] != report.summary["total"] else 0),
    )


__all__ = [
    "ReleaseBuildRequest",
    "ReleaseVerifyRequest",
    "register_release_commands",
]
