from __future__ import annotations

from sol_execbench.cli import evaluation
from sol_execbench.cli import main as cli_main
from sol_execbench.cli import sidecars


def test_cli_main_keeps_sidecar_helper_compatibility_exports() -> None:
    assert cli_main._write_environment_snapshot_sidecar is (
        sidecars._write_environment_snapshot_sidecar
    )
    assert cli_main._write_profile_sidecar is sidecars._write_profile_sidecar
    assert cli_main._write_profile_summary_sidecar is (
        sidecars._write_profile_summary_sidecar
    )
    assert cli_main._write_static_evidence_sidecar is (
        sidecars._write_static_evidence_sidecar
    )
    assert cli_main._write_agent_feedback_sidecar is (
        sidecars._write_agent_feedback_sidecar
    )
    assert cli_main._agent_feedback_identity_fields is (
        sidecars._agent_feedback_identity_fields
    )
    assert cli_main._profile_output_directory is sidecars._profile_output_directory
    assert cli_main._profile_summary_sidecar_path is (
        sidecars._profile_summary_sidecar_path
    )
    assert cli_main._static_evidence_directory is sidecars._static_evidence_directory
    assert cli_main._static_evidence_sidecar_path is (
        sidecars._static_evidence_sidecar_path
    )
    assert (
        cli_main._agent_feedback_sidecar_path is sidecars._agent_feedback_sidecar_path
    )
    assert cli_main._collect_static_evidence_for_cli is not (
        sidecars._collect_static_evidence_for_cli
    )


def test_cli_main_delegates_evaluation_helpers_through_stable_wrappers() -> None:
    assert cli_main._no_trace_diagnostics_sidecar_path is (
        evaluation._no_trace_diagnostics_sidecar_path
    )
    assert cli_main._write_no_trace_diagnostics_sidecar is (
        evaluation._write_no_trace_diagnostics_sidecar
    )
    assert cli_main._timeout_output_text is evaluation._timeout_output_text
    assert cli_main._run_evaluation_command is not evaluation._run_evaluation_command
    assert cli_main._run_profiled_evaluation is not evaluation._run_profiled_evaluation
