from __future__ import annotations

from sol_execbench.cli import evaluation
from sol_execbench.cli import environment
from sol_execbench.cli import main as cli_main
from sol_execbench.cli import reporting
from sol_execbench.cli import sidecars


def test_cli_environment_helpers_live_outside_main_and_sidecars() -> None:
    assert environment._write_environment_snapshot_sidecar is not None
    assert environment._environment_snapshot_sidecar_path is not None
    assert environment.ENV_SNAPSHOT_ENABLE_ENV == "SOLEXECBENCH_ENV_SNAPSHOT"
    assert environment.ENV_SNAPSHOT_PATH_ENV == "SOLEXECBENCH_ENV_SNAPSHOT_PATH"

    for module in (cli_main, sidecars):
        for name in (
            "ENV_SNAPSHOT_ENABLE_ENV",
            "ENV_SNAPSHOT_PATH_ENV",
            "_environment_snapshot_sidecar_path",
            "_write_environment_snapshot_sidecar",
        ):
            assert not hasattr(module, name)


def test_cli_reporting_helpers_live_outside_main() -> None:
    assert reporting.print_traces_table is not None
    assert not hasattr(cli_main, "_print_traces_table")


def test_cli_sidecar_helpers_live_outside_main() -> None:
    assert sidecars._write_profile_sidecar is not None
    assert sidecars._write_profile_summary_sidecar is not None
    assert sidecars._write_static_evidence_sidecar is not None
    assert sidecars._write_agent_feedback_sidecar is not None
    assert sidecars._collect_static_evidence_for_cli is not None

    for name in (
        "_write_profile_sidecar",
        "_write_profile_summary_sidecar",
        "_write_static_evidence_sidecar",
        "_write_agent_feedback_sidecar",
        "_collect_static_evidence_for_cli",
    ):
        assert not hasattr(cli_main, name)


def test_cli_evaluation_helpers_live_outside_main() -> None:
    assert evaluation._write_no_trace_diagnostics_sidecar is not None
    assert evaluation._timeout_output_text is not None
    assert evaluation._run_evaluation_command is not None
    assert evaluation._run_profiled_evaluation is not None

    for name in (
        "_write_no_trace_diagnostics_sidecar",
        "_timeout_output_text",
        "_run_evaluation_command",
        "_run_profiled_evaluation",
    ):
        assert not hasattr(cli_main, name)


def test_cli_metadata_commands_live_outside_main() -> None:
    from sol_execbench.cli import metadata

    assert metadata._contract_cli is not None
    assert metadata._doctor_cli is not None
    assert metadata._toolchain_cli is not None

    for name in (
        "_contract_cli",
        "_doctor_cli",
        "_toolchain_cli",
    ):
        assert not hasattr(cli_main, name)


def test_cli_baseline_commands_live_outside_main() -> None:
    from sol_execbench.cli import baseline

    assert baseline._baseline_cli is not None
    assert baseline._baseline_export_cli is not None

    for name in (
        "_baseline_cli",
        "_baseline_export_cli",
    ):
        assert not hasattr(cli_main, name)


def test_cli_dataset_commands_live_outside_main() -> None:
    from sol_execbench.cli import dataset

    assert dataset._dataset_cli is not None
    assert dataset._dataset_migrate_sol_cli is not None
    assert dataset._dataset_migrate_flashinfer_cli is not None

    for name in (
        "_dataset_cli",
        "_dataset_migrate_sol_cli",
        "_dataset_migrate_flashinfer_cli",
    ):
        assert not hasattr(cli_main, name)


def test_cli_profile_sidecar_helpers_live_outside_main() -> None:
    from sol_execbench.cli import profile_sidecars

    profile_sidecar_names = (
        "_profile_output_directory",
        "_profile_sidecar_path",
        "_write_profile_sidecar",
        "_profile_summary_sidecar_path",
        "_write_profile_summary_sidecar",
        "_profile_summary_artifact_citations",
    )

    for name in profile_sidecar_names:
        assert getattr(profile_sidecars, name) is not None
        assert getattr(sidecars, name) is not None
        assert not hasattr(cli_main, name)


def test_cli_static_evidence_helpers_live_outside_main() -> None:
    from sol_execbench.cli import static_evidence

    assert static_evidence.STATIC_EVIDENCE_NONE == "none"
    assert static_evidence.STATIC_EVIDENCE_AUTO == "auto"
    assert static_evidence._static_evidence_directory is not None
    assert static_evidence._static_evidence_sidecar_path is not None
    assert static_evidence._static_evidence_summary is not None
    assert static_evidence._static_evidence_payload is not None
    assert static_evidence._write_static_evidence_sidecar is not None
    assert static_evidence._collect_static_evidence_for_cli is not None

    for name in (
        "STATIC_EVIDENCE_NONE",
        "STATIC_EVIDENCE_AUTO",
        "_static_evidence_directory",
        "_static_evidence_sidecar_path",
        "_static_evidence_summary",
        "_static_evidence_payload",
        "_write_static_evidence_sidecar",
        "_collect_static_evidence_for_cli",
    ):
        assert not hasattr(cli_main, name)


def test_cli_agent_feedback_sidecar_helpers_live_outside_main() -> None:
    from sol_execbench.cli import agent_feedback_sidecar

    assert agent_feedback_sidecar._agent_feedback_sidecar_path is not None
    assert agent_feedback_sidecar._write_agent_feedback_sidecar is not None
    assert agent_feedback_sidecar._agent_feedback_identity_fields is not None
    assert agent_feedback_sidecar._agent_feedback_run_id is not None
    assert agent_feedback_sidecar._agent_feedback_artifact_citations is not None

    for name in (
        "_agent_feedback_sidecar_path",
        "_write_agent_feedback_sidecar",
        "_agent_feedback_identity_fields",
        "_agent_feedback_run_id",
        "_agent_feedback_artifact_citations",
    ):
        assert not hasattr(cli_main, name)


def test_cli_problem_io_helpers_live_outside_main() -> None:
    from sol_execbench.cli import problem_io

    assert problem_io.ResolvedProblemInputs is not None
    assert problem_io._load_definition is not None
    assert problem_io._load_workloads is not None
    assert problem_io._load_solution is not None
    assert problem_io._load_config is not None
    assert problem_io._resolve_problem_dir is not None
    assert problem_io.resolve_problem_inputs is not None

    for name in (
        "ResolvedProblemInputs",
        "_load_definition",
        "_load_workloads",
        "_load_solution",
        "_load_config",
        "_resolve_problem_dir",
        "resolve_problem_inputs",
    ):
        assert not hasattr(cli_main, name)


def test_cli_compilation_helpers_live_outside_main() -> None:
    from sol_execbench.cli import compilation

    assert compilation.CompilePhaseResult is not None
    assert compilation.run_compile_phase is not None

    for name in (
        "CompilePhaseResult",
        "run_compile_phase",
    ):
        assert not hasattr(cli_main, name)


def test_cli_evaluation_runtime_helpers_live_outside_main() -> None:
    from sol_execbench.cli import evaluation_runtime

    assert evaluation_runtime.EvaluationRuntimeSuccess is not None
    assert evaluation_runtime.EvaluationRuntimeNoTraceFailure is not None
    assert evaluation_runtime.run_evaluation_runtime is not None

    for name in (
        "EvaluationRuntimeSuccess",
        "EvaluationRuntimeNoTraceFailure",
        "run_evaluation_runtime",
    ):
        assert not hasattr(cli_main, name)
