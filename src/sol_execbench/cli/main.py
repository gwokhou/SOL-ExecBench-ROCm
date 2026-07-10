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

import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Optional

import click

import sol_execbench.cli.sidecars.decision as cli_decision
import sol_execbench.cli.sidecars.static_evidence as cli_static_evidence
from sol_execbench.cli.commands import dispatch_subcommand
from sol_execbench.cli.evaluation.evaluator import (
    PROFILE_NONE,
    PROFILE_ROCPROFV3,
    run_evaluation_cli,
)

_COMPILE_PROGRESS_TEXT = "Compiling HIP/C++ solution..."


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
    "--decision",
    type=click.Choice([cli_decision.DECISION_NONE, cli_decision.DECISION_AUTO]),
    default=cli_decision.DECISION_NONE,
    show_default=True,
    help="Emit optional Layer R decision sidecar from static footprints",
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
    decision: str,
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
    run_evaluation_cli(
        problem_dir=problem_dir,
        definition_file=definition_file,
        workload_file=workload_file,
        solution_file=solution_file,
        config_file=config_file,
        compile_timeout=compile_timeout,
        timeout=timeout,
        output_file=output_file,
        json_output=json_output,
        lock_clocks=lock_clocks,
        keep_staging=keep_staging,
        profile=profile,
        static_evidence=static_evidence,
        decision=decision,
        feedback_run_id=feedback_run_id,
        feedback_target_id=feedback_target_id,
        feedback_candidate_id=feedback_candidate_id,
        feedback_source_sha256=feedback_source_sha256,
        feedback_sol_version=feedback_sol_version,
        verbose=verbose,
    )


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
