# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SOL-ExecBench CLI — evaluate solutions locally on GPU.

Usage:
    uv run sol-execbench <problem_dir> --solution solution.json
    uv run sol-execbench --definition def.json --workload wkl.jsonl --solution sol.json
"""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from ..core.data.contract import build_evaluator_contract
from ..core.environment import (
    EnvironmentSnapshot,
    build_environment_diagnostics,
    collect_environment_snapshot,
)
from ..core.bench.rocm_profiler import (
    ROCPROFV3_EXECUTABLE,
    Rocprofv3ProfileRequest,
    Rocprofv3ProfileResult,
    collect_rocprofv3_profile,
)
from ..core import (
    Definition,
    Workload,
    Solution,
    Trace,
    BenchmarkConfig,
    EvaluationStatus,
)
from ..driver import ProblemPackager

console = Console(stderr=True)

ENV_SNAPSHOT_ENABLE_ENV = "SOLEXECBENCH_ENV_SNAPSHOT"
ENV_SNAPSHOT_PATH_ENV = "SOLEXECBENCH_ENV_SNAPSHOT_PATH"
PROFILE_NONE = "none"
PROFILE_ROCPROFV3 = "rocprofv3"


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
    # Resolve source file contents relative to the solution JSON directory.
    sol_dir = path.parent
    for src in sol_dict.get("sources", []):
        if not src.get("content"):
            src_path = sol_dir / src["path"]
            if src_path.exists():
                src["content"] = src_path.read_text()
    return Solution(**sol_dict)


def _load_config(path: Optional[Path]) -> BenchmarkConfig:
    if path is None:
        return BenchmarkConfig()
    return BenchmarkConfig(**json.loads(path.read_text()))


def _resolve_problem_dir(
    problem_dir: Path,
) -> tuple[Path, Path, Optional[Path], Optional[Path]]:
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


def _print_traces_table(traces: list[Trace]) -> None:
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

    # Show logs for traces with runtime errors
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
        if isinstance(snapshot, EnvironmentSnapshot):
            payload = snapshot.model_dump(mode="json")
        elif hasattr(snapshot, "model_dump"):
            payload = snapshot.model_dump(mode="json")
        else:
            payload = snapshot
        sidecar_path.parent.mkdir(parents=True, exist_ok=True)
        sidecar_path.write_text(json.dumps(payload, sort_keys=True) + "\n")
        console.print(f"[green]Saved environment snapshot to {sidecar_path}[/green]")
        return sidecar_path
    except Exception as exc:
        console.print(f"[yellow]Environment snapshot skipped: {exc}[/yellow]")
        return None


def _profile_output_directory(output_file: Path | None, staging_dir: Path) -> Path:
    """Return the profiler artifact directory for an evaluation run."""

    if output_file is not None:
        return output_file.with_name(f"{output_file.name}.rocprofv3")
    return staging_dir / "rocprofv3"


def _profile_sidecar_path(
    output_file: Path | None,
    profile_result: Rocprofv3ProfileResult,
) -> Path:
    """Return the profile metadata sidecar path for an evaluation run."""

    if output_file is not None:
        return output_file.with_name(f"{output_file.name}.profile.json")
    return profile_result.output_directory / "profile.json"


def _write_profile_sidecar(
    output_file: Path | None,
    profile_result: Rocprofv3ProfileResult | None,
) -> Path | None:
    """Write optional profiler metadata without changing trace JSONL."""

    if profile_result is None:
        return None

    sidecar_path = _profile_sidecar_path(output_file, profile_result)
    try:
        sidecar_path.parent.mkdir(parents=True, exist_ok=True)
        sidecar_path.write_text(
            json.dumps(profile_result.to_dict(), sort_keys=True) + "\n"
        )
        console.print(f"[green]Saved profiling metadata to {sidecar_path}[/green]")
        return sidecar_path
    except Exception as exc:
        console.print(f"[yellow]Profiling metadata skipped: {exc}[/yellow]")
        return None


def _run_evaluation_command(
    eval_cmd: list[str],
    *,
    staging_dir: Path,
    timeout: int,
) -> subprocess.CompletedProcess[str]:
    """Run the staged evaluation command with the standard ROCm allocator env."""

    return subprocess.run(
        eval_cmd,
        cwd=staging_dir,
        capture_output=True,
        text=True,
        timeout=timeout,
        env={**os.environ, "PYTORCH_ALLOC_CONF": "expandable_segments:True"},
    )


def _run_profiled_evaluation(
    eval_cmd: list[str],
    *,
    staging_dir: Path,
    output_file: Path | None,
    timeout: int,
) -> tuple[subprocess.CompletedProcess[str] | None, Rocprofv3ProfileResult]:
    """Run evaluation under `rocprofv3`, returning normal execution on failure."""

    output_directory = _profile_output_directory(output_file, staging_dir)
    request = Rocprofv3ProfileRequest(
        application_command=tuple(eval_cmd),
        output_directory=output_directory,
        output_file="profile",
        working_directory=staging_dir,
        timeout_seconds=timeout,
    )
    profile_result = collect_rocprofv3_profile(
        request,
        rocprofv3_available=shutil.which(ROCPROFV3_EXECUTABLE) is not None,
        runner=lambda command, cwd, timeout_seconds: subprocess.run(
            list(command),
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env={**os.environ, "PYTORCH_ALLOC_CONF": "expandable_segments:True"},
        ),
    )
    if profile_result.succeeded:
        profiled_proc = subprocess.CompletedProcess(
            args=list(profile_result.command),
            returncode=profile_result.returncode or 0,
            stdout=profile_result.stdout,
            stderr=profile_result.stderr,
        )
        return profiled_proc, profile_result
    return None, profile_result


@click.command(
    name="sol-execbench", context_settings={"help_option_names": ["-h", "--help"]}
)
@click.argument(
    "problem_dir", required=False, type=click.Path(exists=True, path_type=Path)
)
@click.option(
    "--definition",
    "definition_file",
    type=click.Path(exists=True, path_type=Path),
    help="Path to definition.json",
)
@click.option(
    "--workload",
    "workload_file",
    type=click.Path(exists=True, path_type=Path),
    help="Path to workload.jsonl",
)
@click.option(
    "--solution",
    "solution_file",
    type=click.Path(exists=True, path_type=Path),
    help="Path to solution.json",
)
@click.option(
    "--config",
    "config_file",
    type=click.Path(exists=True, path_type=Path),
    help="Path to benchmark config JSON",
)
@click.option(
    "--compile-timeout",
    default=120,
    type=int,
    help="Compilation timeout in seconds (HIP/C++ only)",
)
@click.option(
    "--timeout", default=600, type=int, help="Evaluation subprocess timeout in seconds"
)
@click.option(
    "-o",
    "--output",
    "output_file",
    type=click.Path(path_type=Path),
    help="Write trace JSONL to this file",
)
@click.option("--json", "json_output", is_flag=True, help="Print trace JSON to stdout")
@click.option("--lock-clocks", is_flag=True, help="Require GPU clocks to be locked")
@click.option(
    "--keep-staging", is_flag=True, help="Keep the staging directory after evaluation"
)
@click.option(
    "--profile",
    type=click.Choice([PROFILE_NONE, PROFILE_ROCPROFV3]),
    default=PROFILE_NONE,
    show_default=True,
    help="Collect optional diagnostic profiling artifacts",
)
@click.option("--verbose", "-v", is_flag=True, help="Show subprocess output")
def _evaluate_cli(
    problem_dir: Optional[Path],
    definition_file: Optional[Path],
    workload_file: Optional[Path],
    solution_file: Path,
    config_file: Optional[Path],
    compile_timeout: int,
    timeout: int,
    output_file: Optional[Path],
    json_output: bool,
    lock_clocks: bool,
    keep_staging: bool,
    profile: str,
    verbose: bool,
):
    """Evaluate a SOL-ExecBench solution on GPU.

    \b
    Two ways to specify the problem:
      1) Positional: sol-execbench <problem_dir> --solution sol.json
         (reads definition.json and workload.jsonl from problem_dir)
      2) Explicit:   sol-execbench --definition def.json --workload wkl.jsonl --solution sol.json

    \b
    Metadata:
      sol-execbench contract --json
    """
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

    # Load data models
    definition = _load_definition(definition_file)
    workloads = _load_workloads(workload_file)
    solution = _load_solution(solution_file)
    config = _load_config(config_file)

    if lock_clocks:
        config.lock_clocks = True

    console.print(f"[bold]Problem:[/bold]  {definition.name}")
    console.print(f"[bold]Solution:[/bold] {solution.name}")
    console.print(f"[bold]Workloads:[/bold] {len(workloads)}")
    if config_file:
        console.print(
            f"[bold]Config:[/bold]   {json.dumps(dataclasses.asdict(config))}"
        )

    # Create staging directory
    staging_dir = Path(tempfile.mkdtemp(prefix="sol_execbench_"))
    packager = ProblemPackager(
        definition=definition,
        workloads=workloads,
        solution=solution,
        config=config,
        output_dir=staging_dir,
        keep_output_dir=keep_staging,
    )

    if verbose:
        console.print(f"[dim]Staging dir: {staging_dir}[/dim]")

    # Phase 1: Compile (HIP/C++ only)
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
                env={**os.environ, "PYTORCH_ALLOC_CONF": "expandable_segments:True"},
            )
            progress.update(task, completed=True)

        if proc.returncode != 0:
            console.print("[red]Compilation failed[/red]")
            if proc.stderr:
                console.print(proc.stderr)
            if proc.stdout:
                console.print(proc.stdout)
            sys.exit(1)

        console.print("[green]Compilation succeeded[/green]")
        if verbose and proc.stderr:
            console.print(f"[dim]{proc.stderr}[/dim]")

    # Phase 2: Evaluate
    eval_cmd = packager.execute()

    profile_result: Rocprofv3ProfileResult | None = None
    profiled_proc: subprocess.CompletedProcess[str] | None = None
    if profile == PROFILE_ROCPROFV3:
        console.print("[dim]Collecting optional rocprofv3 profiling evidence...[/dim]")
        profiled_proc, profile_result = _run_profiled_evaluation(
            eval_cmd,
            staging_dir=staging_dir,
            output_file=output_file,
            timeout=timeout,
        )
        if profiled_proc is None:
            reason = profile_result.skipped_reason or profile_result.failed_reason
            console.print(
                "[yellow]rocprofv3 profiling unavailable or failed; "
                f"running normal evaluation. Reason: {reason}[/yellow]"
            )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(
            f"Evaluating {len(workloads)} workload(s)...", total=None
        )

        proc = profiled_proc or _run_evaluation_command(
            eval_cmd,
            staging_dir=staging_dir,
            timeout=timeout,
        )
        progress.update(task, completed=True)

    if verbose and proc.stderr:
        console.print(f"[dim]{proc.stderr}[/dim]")

    if proc.returncode != 0 and not proc.stdout.strip():
        console.print("[red]Evaluation failed[/red]")
        if proc.stderr:
            console.print(proc.stderr)
        sys.exit(1)

    # Parse traces from stdout
    traces = packager.convert_stdout_to_traces(proc.stdout)

    if not traces:
        console.print("[red]No traces produced[/red]")
        if proc.stderr:
            console.print(proc.stderr)
        sys.exit(1)

    # Output
    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            for t in traces:
                f.write(json.dumps(t.model_dump(mode="json")) + "\n")
        console.print(f"[green]Saved {len(traces)} traces to {output_file}[/green]")

    _write_environment_snapshot_sidecar(output_file)
    _write_profile_sidecar(output_file, profile_result)

    if json_output:
        for t in traces:
            print(json.dumps(t.model_dump(mode="json")))
    else:
        _print_traces_table(traces)

    # Exit code: 0 if all passed, 1 otherwise
    all_passed = all(
        t.evaluation and t.evaluation.status == EvaluationStatus.PASSED for t in traces
    )
    sys.exit(0 if all_passed else 1)


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


class SolExecbenchCli(click.Command):
    """Dispatch root evaluator calls and the contract metadata command."""

    def main(
        self,
        args: list[str] | None = None,
        prog_name: str | None = None,
        complete_var: str | None = None,
        standalone_mode: bool = True,
        windows_expand_args: bool = True,
        **extra: object,
    ) -> object:
        args = list(args) if args is not None else sys.argv[1:]
        if args and args[0] == "contract":
            contract_prog = f"{prog_name or self.name} contract"
            return _contract_cli.main(
                args=args[1:],
                prog_name=contract_prog,
                complete_var=complete_var,
                standalone_mode=standalone_mode,
                windows_expand_args=windows_expand_args,
                **extra,
            )
        if args and args[0] == "doctor":
            doctor_prog = f"{prog_name or self.name} doctor"
            return _doctor_cli.main(
                args=args[1:],
                prog_name=doctor_prog,
                complete_var=complete_var,
                standalone_mode=standalone_mode,
                windows_expand_args=windows_expand_args,
                **extra,
            )
        return _evaluate_cli.main(
            args=args,
            prog_name=prog_name or self.name,
            complete_var=complete_var,
            standalone_mode=standalone_mode,
            windows_expand_args=windows_expand_args,
            **extra,
        )


cli = SolExecbenchCli(
    name="sol-execbench",
    help="Evaluate solutions or print GPU-free evaluator contract metadata.",
)


if __name__ == "__main__":
    cli()
