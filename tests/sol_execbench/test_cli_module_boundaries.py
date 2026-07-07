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
