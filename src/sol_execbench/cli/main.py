# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
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
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Optional

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from . import baseline as cli_baseline
from . import agent_feedback_sidecar as cli_agent_feedback_sidecar
from . import dataset as cli_dataset
from . import evaluation as cli_evaluation
from . import environment as cli_environment
from . import metadata as cli_metadata
from . import profile_sidecars as cli_profile_sidecars
from . import reporting as cli_reporting
from . import static_evidence as cli_static_evidence
from ..core.bench.io import flashinfer_safetensors_env
from ..core.bench.rocm_profiler import Rocprofv3ProfileResult
from ..core.bench.stderr import filter_benign_rocm_stderr
from ..core.bench.static_kernel_evidence import StaticKernelEvidenceSidecar
from ..core import (
    Definition,
    Workload,
    Solution,
    BenchmarkConfig,
    EvaluationStatus,
)
from ..core.dataset.checksums import sha256_file
from ..driver import ProblemPackager

console = Console(stderr=True)

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
@click.option(
    "--static-evidence",
    type=click.Choice(
        [
            cli_static_evidence.STATIC_EVIDENCE_NONE,
            cli_static_evidence.STATIC_EVIDENCE_AUTO,
        ]
    ),
    default=cli_static_evidence.STATIC_EVIDENCE_NONE,
    show_default=True,
    help="Collect optional diagnostic static kernel evidence",
)
@click.option(
    "--feedback-target-id",
    help="Consumer target identity to persist in diagnostic agent feedback.",
)
@click.option(
    "--feedback-run-id",
    help="Consumer run identity to persist in diagnostic agent feedback.",
)
@click.option(
    "--feedback-candidate-id",
    help="Consumer candidate identity to persist in diagnostic agent feedback.",
)
@click.option(
    "--feedback-source-sha256",
    help="Consumer source SHA256 identity to persist in diagnostic agent feedback.",
)
@click.option(
    "--feedback-sol-version",
    help="Consumer SOL version/tag identity to persist in diagnostic agent feedback.",
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
    static_evidence: str,
    feedback_target_id: str | None,
    feedback_run_id: str | None,
    feedback_candidate_id: str | None,
    feedback_source_sha256: str | None,
    feedback_sol_version: str | None,
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
            static_evidence_result = (
                cli_static_evidence._collect_static_evidence_for_cli(
                    enabled=static_evidence,
                    is_cpp=True,
                    staging_dir=staging_dir,
                    output_file=output_file,
                )
            )
    elif static_evidence == cli_static_evidence.STATIC_EVIDENCE_AUTO:
        static_evidence_result = cli_static_evidence._collect_static_evidence_for_cli(
            enabled=static_evidence,
            is_cpp=False,
            staging_dir=staging_dir,
            output_file=output_file,
        )

    # Phase 2: Evaluate
    eval_cmd = packager.execute()

    profile_result: Rocprofv3ProfileResult | None = None
    profiled_proc: subprocess.CompletedProcess[str] | None = None
    if profile == PROFILE_ROCPROFV3:
        console.print("[dim]Collecting optional rocprofv3 profiling evidence...[/dim]")
        profiled_proc, profile_result = cli_evaluation._run_profiled_evaluation(
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

        try:
            proc = profiled_proc or cli_evaluation._run_evaluation_command(
                eval_cmd,
                staging_dir=staging_dir,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            # subprocess.run raises on timeout instead of returning a
            # CompletedProcess; synthesize the no-trace diagnostic path so a
            # hung/deadlocked evaluation produces a clean sidecar + exit rather
            # than an unhandled traceback.
            progress.update(task, completed=True)
            console.print(f"[red]Evaluation timed out after {timeout}s[/red]")
            diagnostic_path = cli_evaluation._write_no_trace_diagnostics_sidecar(
                output_file=output_file,
                staging_dir=staging_dir,
                keep_staging=keep_staging,
                reason="evaluation_timeout",
                returncode=124,
                stdout=cli_evaluation._timeout_output_text(exc.stdout),
                stderr=cli_evaluation._timeout_output_text(exc.stderr),
            )
            if diagnostic_path is not None:
                console.print(
                    f"[yellow]Saved no-trace diagnostics to {diagnostic_path}[/yellow]"
                )
            packager.close()
            sys.exit(1)
        progress.update(task, completed=True)

    filtered_stderr = filter_benign_rocm_stderr(proc.stderr)
    if verbose and filtered_stderr:
        console.print(f"[dim]{filtered_stderr}[/dim]")

    if proc.returncode != 0 and not proc.stdout.strip():
        console.print("[red]Evaluation failed[/red]")
        diagnostic_path = cli_evaluation._write_no_trace_diagnostics_sidecar(
            output_file=output_file,
            staging_dir=staging_dir,
            keep_staging=keep_staging,
            reason="evaluation_failed_no_stdout",
            returncode=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
        )
        if diagnostic_path is not None:
            console.print(
                f"[yellow]Saved no-trace diagnostics to {diagnostic_path}[/yellow]"
            )
        if filtered_stderr:
            console.print(filtered_stderr)
        packager.close()
        sys.exit(1)

    # Parse traces from stdout
    traces = packager.convert_stdout_to_traces(proc.stdout)

    if not traces:
        console.print("[red]No traces produced[/red]")
        diagnostic_path = cli_evaluation._write_no_trace_diagnostics_sidecar(
            output_file=output_file,
            staging_dir=staging_dir,
            keep_staging=keep_staging,
            reason="no_parseable_traces",
            returncode=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
        )
        if diagnostic_path is not None:
            console.print(
                f"[yellow]Saved no-trace diagnostics to {diagnostic_path}[/yellow]"
            )
        if filtered_stderr:
            console.print(filtered_stderr)
        packager.close()
        sys.exit(1)

    # Output
    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            for t in traces:
                f.write(json.dumps(t.model_dump(mode="json")) + "\n")
        console.print(f"[green]Saved {len(traces)} traces to {output_file}[/green]")

    trace_run_id = (
        sha256_file(output_file)
        if output_file is not None and output_file.is_file()
        else None
    )
    environment_sidecar_path = cli_environment._write_environment_snapshot_sidecar(
        output_file
    )
    profile_sidecar_path = cli_profile_sidecars._write_profile_sidecar(
        output_file, profile_result
    )
    cli_profile_sidecars._write_profile_summary_sidecar(
        output_file,
        profile_result,
        profile_sidecar_path=profile_sidecar_path,
        run_id=trace_run_id,
    )
    static_evidence_sidecar_path = cli_static_evidence._write_static_evidence_sidecar(
        output_file,
        staging_dir,
        static_evidence_result,
    )
    cli_agent_feedback_sidecar._write_agent_feedback_sidecar(
        output_file,
        traces,
        solution=solution,
        profile_result=profile_result,
        static_evidence=static_evidence_result,
        environment_sidecar_path=environment_sidecar_path,
        profile_sidecar_path=profile_sidecar_path,
        static_evidence_sidecar_path=static_evidence_sidecar_path,
        run_id=feedback_run_id or trace_run_id,
        feedback_target_id=feedback_target_id,
        feedback_candidate_id=feedback_candidate_id,
        feedback_source_sha256=feedback_source_sha256,
        feedback_sol_version=feedback_sol_version,
    )

    if json_output:
        for t in traces:
            print(json.dumps(t.model_dump(mode="json")))
    else:
        cli_reporting.print_traces_table(traces)

    packager.close()

    # Exit code: 0 if all passed, 1 otherwise
    all_passed = all(
        t.evaluation and t.evaluation.status == EvaluationStatus.PASSED for t in traces
    )
    sys.exit(0 if all_passed else 1)


class SolExecbenchCli(click.Command):
    """Dispatch root evaluator calls and the contract metadata command."""

    def main(
        self,
        args: Sequence[str] | None = None,
        prog_name: str | None = None,
        complete_var: str | None = None,
        standalone_mode: bool = True,
        windows_expand_args: bool = True,
        **extra: Any,
    ) -> Any:
        args = list(args) if args is not None else sys.argv[1:]
        if args and args[0] == "contract":
            contract_prog = f"{prog_name or self.name} contract"
            return cli_metadata._contract_cli.main(
                args=args[1:],
                prog_name=contract_prog,
                complete_var=complete_var,
                standalone_mode=standalone_mode,
                windows_expand_args=windows_expand_args,
                **extra,
            )
        if args and args[0] == "doctor":
            doctor_prog = f"{prog_name or self.name} doctor"
            return cli_metadata._doctor_cli.main(
                args=args[1:],
                prog_name=doctor_prog,
                complete_var=complete_var,
                standalone_mode=standalone_mode,
                windows_expand_args=windows_expand_args,
                **extra,
            )
        if args and args[0] == "toolchain":
            toolchain_prog = f"{prog_name or self.name} toolchain"
            return cli_metadata._toolchain_cli.main(
                args=args[1:],
                prog_name=toolchain_prog,
                complete_var=complete_var,
                standalone_mode=standalone_mode,
                windows_expand_args=windows_expand_args,
                **extra,
            )
        if args and args[0] == "baseline":
            baseline_prog = f"{prog_name or self.name} baseline"
            return cli_baseline._baseline_cli.main(
                args=args[1:],
                prog_name=baseline_prog,
                complete_var=complete_var,
                standalone_mode=standalone_mode,
                windows_expand_args=windows_expand_args,
                **extra,
            )
        if args and args[0] == "dataset":
            dataset_prog = f"{prog_name or self.name} dataset"
            return cli_dataset._dataset_cli.main(
                args=args[1:],
                prog_name=dataset_prog,
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
    help="Evaluate solutions or print GPU-free evaluator/toolchain metadata.",
)


if __name__ == "__main__":
    cli()
