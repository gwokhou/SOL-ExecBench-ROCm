# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Click commands for the ROCm vertical 3-tier SOL diagnosis suite."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import click

from sol_execbench.cli.protocol import CliExitCode, CliResult, artifact
from sol_execbench.vertical.cloud_preflight import (
    build_oneshot_cloud_script,
    probe_remote_connectivity,
    validate_downloaded_bundle,
    verify_python312_twin,
)
from sol_execbench.vertical.cloud_runner import run_benchmark_campaign
from sol_execbench.vertical.kernels import KernelBackend
from sol_execbench.vertical.report import compare_summaries_markdown
from sol_execbench.vertical.workloads import SweepMode


@click.group(
    "vertical",
    context_settings={"help_option_names": ["-h", "--help"]},
)
def vertical_cli() -> None:
    """Run ROCm vertical 3-tier analytical SOL diagnosis and preflight gates."""


@vertical_cli.command("run")
@click.option(
    "--mode",
    type=click.Choice([m.value for m in SweepMode]),
    default=SweepMode.SMOKE.value,
    show_default=True,
)
@click.option("--device", default="auto", show_default=True)
@click.option(
    "--backend",
    type=click.Choice([b.value for b in KernelBackend]),
    default=KernelBackend.AUTO.value,
    show_default=True,
)
@click.option("--target-override", default=None)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path("data/outputs/vertical_sol"),
    show_default=True,
)
@click.option(
    "--collect-counters/--no-collect-counters",
    default=False,
    show_default=True,
)
@click.option(
    "--lock-clocks/--no-lock-clocks",
    default=False,
    show_default=True,
)
@click.option("--workload-filter", default=None)
def vertical_run_cli(
    mode: str,
    device: str,
    backend: str,
    target_override: str | None,
    output_dir: Path,
    collect_counters: bool,
    lock_clocks: bool,
    workload_filter: str | None,
) -> CliResult:
    """Run the anti-distortion ROCm vertical SOL campaign."""
    written = run_benchmark_campaign(
        mode=SweepMode(mode),
        device=device,
        target_override=target_override,
        backend=KernelBackend(backend),
        output_dir=output_dir,
        collect_counters=collect_counters,
        lock_clocks=lock_clocks,
        workload_filter=workload_filter,
    )
    click.echo(f"Report: {written['report_md']}")
    click.echo(f"Summary: {written['summary_json']}")
    return CliResult(
        data={k: str(v) for k, v in written.items()},
        artifacts=(
            artifact(written["report_md"], "report_md"),
            artifact(written["summary_json"], "summary_json"),
            artifact(written["traces_jsonl"], "traces_jsonl"),
        ),
    )


@vertical_cli.command("compare")
@click.argument(
    "paths",
    nargs=-1,
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option("--label", "labels", multiple=True)
def vertical_compare_cli(
    paths: tuple[Path, ...],
    labels: tuple[str, ...],
) -> CliResult:
    """Compare 2 or 3 vertical SOL summary/report files side by side."""
    markdown = compare_summaries_markdown(
        *paths,
        labels=labels or None,
    )
    click.echo(markdown, nl=False)
    return CliResult(
        data={"markdown": markdown, "paths": [str(p) for p in paths]}
    )


@vertical_cli.group(
    "preflight",
    context_settings={"help_option_names": ["-h", "--help"]},
)
def vertical_preflight_cli() -> None:
    """Run cloud preflight twin, connectivity probe, bundle, and verify gates."""


@vertical_preflight_cli.command("twin")
@click.option("--target", default="MI300X", show_default=True)
def preflight_twin_cli(target: str) -> CliResult:
    """Verify native Python 3.12 syntax and uncached CPU smoke execution."""
    result = verify_python312_twin(target_id=target)
    payload = asdict(result)
    click.echo(
        f"Twin passed for {result.target_id}: "
        f"{result.checked_modules} modules, {result.smoke_rows} smoke rows"
    )
    return CliResult(
        data=payload,
        exit_code=(
            CliExitCode.SUCCESS if result.passed else CliExitCode.RESULT_FAILED
        ),
    )


@vertical_preflight_cli.command("probe")
@click.argument("host")
@click.option("--ssh-port", type=int, default=22, show_default=True)
@click.option("--http-port", type=int, default=80, show_default=True)
@click.option("--proxy", default=None)
@click.option("--timeout", type=float, default=3.0, show_default=True)
def preflight_probe_cli(
    host: str,
    ssh_port: int,
    http_port: int,
    proxy: str | None,
    timeout: float,
) -> CliResult:
    """Classify remote cloud SSH and HTTP reachability in a single pass."""
    report = probe_remote_connectivity(
        host,
        ssh_port=ssh_port,
        http_port=http_port,
        proxy=proxy,
        timeout_s=timeout,
    )
    payload = asdict(report)
    click.echo(f"{report.host}: {report.verdict.value} ({report.detail})")
    return CliResult(data=payload)


@vertical_preflight_cli.command("bundle")
@click.option(
    "--output",
    required=True,
    type=click.Path(dir_okay=False, path_type=Path),
)
@click.option(
    "--mode",
    type=click.Choice([m.value for m in SweepMode]),
    default=SweepMode.FULL.value,
    show_default=True,
)
@click.option(
    "--remote-out-dir",
    default="/tmp/mvp_cloud_out",
    show_default=True,
)
def preflight_bundle_cli(
    output: Path,
    mode: str,
    remote_out_dir: str,
) -> CliResult:
    """Generate a self-contained one-shot cloud execution bootstrap script."""
    script_path = build_oneshot_cloud_script(
        output,
        mode=SweepMode(mode),
        remote_out_dir=remote_out_dir,
    )
    click.echo(f"Wrote one-shot script: {script_path}")
    return CliResult(
        data={"script": str(script_path)},
        artifacts=(artifact(script_path, "oneshot_script"),),
    )


@vertical_preflight_cli.command("verify")
@click.argument("bundle_path", type=click.Path(exists=True, path_type=Path))
@click.option("--expected-sha256", default=None)
def preflight_verify_cli(
    bundle_path: Path,
    expected_sha256: str | None,
) -> CliResult:
    """Validate SHA-256 digest and FlashInfer-Trace contracts of a bundle."""
    receipt = validate_downloaded_bundle(
        bundle_path,
        expected_sha256=expected_sha256,
    )
    payload = asdict(receipt)
    click.echo(
        f"Verified {receipt.target_id}: {receipt.row_count} rows, "
        f"all_passed={receipt.all_passed}"
    )
    return CliResult(
        data=payload,
        exit_code=(
            CliExitCode.SUCCESS
            if receipt.all_passed
            else CliExitCode.RESULT_FAILED
        ),
    )


__all__ = ["vertical_cli"]
