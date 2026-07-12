"""The explicit ``evaluate`` command."""

from __future__ import annotations

import re
from pathlib import Path

import click

import sol_execbench.cli.sidecars.decision as cli_decision
import sol_execbench.cli.sidecars.static_evidence as cli_static_evidence
from sol_execbench.cli.evaluation.evaluator import (
    PROFILE_NONE,
    PROFILE_ROCPROFV3,
    run_evaluation_cli,
)
from sol_execbench.cli.evaluation.requests import EvaluationRequest
from sol_execbench.cli.protocol import CliFailure, output_format


@click.command("evaluate", context_settings={"help_option_names": ["-h", "--help"]})
@click.argument(
    "problem_dir",
    required=False,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option(
    "--definition",
    "definition_file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--workload",
    "workload_file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--solution",
    "solution_file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--config",
    "config_file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--compile-timeout",
    default=120,
    show_default=True,
    type=click.IntRange(min=1),
    help="Compilation timeout in seconds (HIP/C++ only).",
)
@click.option("--timeout", default=600, show_default=True, type=click.IntRange(min=1))
@click.option(
    "--trace-output",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Write canonical Trace JSONL here.",
)
@click.option("--lock-clocks", is_flag=True)
@click.option("--keep-staging", is_flag=True)
@click.option(
    "--profile",
    type=click.Choice([PROFILE_NONE, PROFILE_ROCPROFV3]),
    default=PROFILE_NONE,
    show_default=True,
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
)
@click.option(
    "--decision",
    type=click.Choice([cli_decision.DECISION_NONE, cli_decision.DECISION_AUTO]),
    default=cli_decision.DECISION_NONE,
    show_default=True,
)
@click.option("--feedback-target-id")
@click.option("--feedback-run-id")
@click.option("--feedback-candidate-id")
@click.option("--feedback-source-sha256")
@click.option("--feedback-sol-version")
@click.option("--release-bound-sha256", callback=lambda c, p, v: _sha256(c, p, v))
@click.option(
    "--release-hardware-model-sha256", callback=lambda c, p, v: _sha256(c, p, v)
)
@click.option(
    "--release-authority-json",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option("--verbose", "verbose", is_flag=True)
def evaluate_cli(
    problem_dir: Path | None,
    definition_file: Path | None,
    workload_file: Path | None,
    solution_file: Path | None,
    config_file: Path | None,
    compile_timeout: int,
    timeout: int,
    trace_output: Path | None,
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
    release_bound_sha256: str | None,
    release_hardware_model_sha256: str | None,
    release_authority_json: Path | None,
    verbose: bool,
):
    """Evaluate a solution and preserve canonical traces as a JSONL artifact.

    Example: ``sol-execbench evaluate PROBLEM_DIR --solution solution.json``

    Without PROBLEM_DIR, --definition, --workload, and --solution are all
    required. JSON response mode additionally requires --trace-output.
    Exit status is 0 when all workloads pass, 1 for a valid non-passing result,
    2 for invalid input, 3 when the environment is unavailable, and 4 on an
    execution failure.
    """
    if problem_dir is not None and (definition_file or workload_file):
        raise CliFailure(
            "PROBLEM_DIR cannot be combined with --definition or --workload",
            code="conflicting_inputs",
        )
    if problem_dir is None and not all((definition_file, workload_file, solution_file)):
        raise CliFailure(
            "without PROBLEM_DIR, --definition, --workload, and --solution are required together",
            code="incomplete_input_set",
        )
    if problem_dir is not None and solution_file is None:
        solution_file = problem_dir / "solution.json"
        if not solution_file.is_file():
            raise CliFailure(
                "--solution is required because PROBLEM_DIR has no solution.json"
            )
    hashes = (release_bound_sha256, release_hardware_model_sha256)
    if release_authority_json is not None and any(hashes):
        raise CliFailure(
            "--release-authority-json is mutually exclusive with release SHA-256 options",
            code="conflicting_inputs",
        )
    if any(hashes) and not all(hashes):
        raise CliFailure(
            "--release-bound-sha256 and --release-hardware-model-sha256 must be supplied together",
            code="incomplete_input_set",
        )
    if output_format() == "json" and trace_output is None:
        raise CliFailure(
            "evaluate in JSON mode requires --trace-output",
            code="missing_trace_output",
            hint="Add --trace-output PATH after the evaluate subcommand.",
        )
    assert solution_file is not None
    request = EvaluationRequest(
        problem_dir=problem_dir,
        definition_file=definition_file,
        workload_file=workload_file,
        solution_file=solution_file,
        config_file=config_file,
        compile_timeout=compile_timeout,
        timeout=timeout,
        output_file=trace_output,
        json_output=False,
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
        release_bound_sha256=release_bound_sha256,
        release_hardware_model_sha256=release_hardware_model_sha256,
        release_authority_json=release_authority_json,
        verbose=verbose,
    )
    return run_evaluation_cli(request=request)


setattr(
    evaluate_cli,
    "cli_constraints",
    [
        "PROBLEM_DIR excludes --definition and --workload",
        "without PROBLEM_DIR, --definition/--workload/--solution are all required",
        "JSON mode requires --trace-output",
        "release authority JSON excludes the paired release SHA-256 options",
        "--release-bound-sha256 and --release-hardware-model-sha256 are all-or-none",
    ],
)


def _sha256(
    ctx: click.Context, param: click.Parameter, value: str | None
) -> str | None:
    if value is not None and re.fullmatch(r"[0-9a-fA-F]{64}", value) is None:
        raise click.BadParameter(
            "must be exactly 64 hexadecimal characters", ctx, param
        )
    return value.lower() if value is not None else None
