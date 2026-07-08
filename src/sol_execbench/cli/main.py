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
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Optional

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from . import agent_feedback_sidecar as cli_agent_feedback_sidecar
from . import compilation as cli_compilation
from . import evaluation as cli_evaluation
from . import environment as cli_environment
from . import evaluation_runtime as cli_evaluation_runtime
from . import problem_io as cli_problem_io
from . import profile_sidecars as cli_profile_sidecars
from . import reporting as cli_reporting
from . import static_evidence as cli_static_evidence
from .commands import dispatch_subcommand
from ..core.bench.static_kernel_evidence import StaticKernelEvidenceSidecar
from ..core import EvaluationStatus
from ..core.checksums import sha256_file
from ..driver import ProblemPackager

console = Console(stderr=True)

PROFILE_NONE = "none"
PROFILE_ROCPROFV3 = "rocprofv3"


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

    # Load data models
    definition = cli_problem_io._load_definition(definition_file)
    workloads = cli_problem_io._load_workloads(workload_file)
    solution = cli_problem_io._load_solution(solution_file)
    config = cli_problem_io._load_config(config_file)

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

            compile_result = cli_compilation.run_compile_phase(
                packager,
                staging_dir=staging_dir,
                compile_timeout=compile_timeout,
            )
            progress.update(task, completed=True)

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

    if profile == PROFILE_ROCPROFV3:
        console.print("[dim]Collecting optional rocprofv3 profiling evidence...[/dim]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(
            f"Evaluating {len(workloads)} workload(s)...", total=None
        )

        runtime_result = cli_evaluation_runtime.run_evaluation_runtime(
            packager,
            eval_cmd=eval_cmd,
            staging_dir=staging_dir,
            output_file=output_file,
            timeout=timeout,
            profile=profile,
        )
        progress.update(task, completed=True)

    profile_result = runtime_result.profile_result
    if runtime_result.profile_fallback_reason is not None:
        console.print(
            "[yellow]rocprofv3 profiling unavailable or failed; "
            "running normal evaluation. Reason: "
            f"{runtime_result.profile_fallback_reason}[/yellow]"
        )

    if isinstance(
        runtime_result, cli_evaluation_runtime.EvaluationRuntimeNoTraceFailure
    ):
        console.print(f"[red]{runtime_result.message}[/red]")
        diagnostic_path = cli_evaluation._write_no_trace_diagnostics_sidecar(
            output_file=output_file,
            staging_dir=staging_dir,
            keep_staging=keep_staging,
            reason=runtime_result.reason,
            returncode=runtime_result.returncode,
            stdout=runtime_result.stdout,
            stderr=runtime_result.stderr,
        )
        if diagnostic_path is not None:
            console.print(
                f"[yellow]Saved no-trace diagnostics to {diagnostic_path}[/yellow]"
            )
        if (
            runtime_result.reason != "evaluation_timeout"
            and runtime_result.filtered_stderr
        ):
            console.print(runtime_result.filtered_stderr)
        packager.close()
        sys.exit(1)

    if verbose and runtime_result.filtered_stderr:
        console.print(f"[dim]{runtime_result.filtered_stderr}[/dim]")

    traces = runtime_result.traces

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
    """Dispatch root evaluator calls and GPU-free metadata subcommands."""

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
        subcommand_dispatch = dispatch_subcommand(
            args,
            root_command=self,
            prog_name=prog_name,
            complete_var=complete_var,
            standalone_mode=standalone_mode,
            windows_expand_args=windows_expand_args,
            extra=extra,
        )
        if subcommand_dispatch is not None:
            return subcommand_dispatch.result

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
