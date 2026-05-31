from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

from sol_execbench.cli.main import cli
from sol_execbench.core.compatibility import (
    ROCM_COMPATIBILITY_MATRIX_SCHEMA_VERSION,
    MatrixArtifactReference,
    MatrixClaimBoundary,
    MatrixCompatibilityReasonCode,
    MatrixCompatibilityStatus,
    MatrixContainerEvidence,
    MatrixDependencyPolicyEvidence,
    MatrixGpuEvidence,
    MatrixHostEvidence,
    MatrixObservedEvidence,
    MatrixPythonDependencyEvidence,
    MatrixTarget,
    MatrixToolchainEvidence,
    MatrixValidationScope,
    RocmCompatibilityMatrixReport,
    build_matrix_entry,
)
from sol_execbench.core.matrix_diff import (
    MatrixDiffSeverity,
    diff_matrix_reports,
    load_matrix_report,
    matrix_report_diff_to_markdown,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "diff_matrix_reports.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("diff_matrix_reports", SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _target(
    target_id: str,
    *,
    scope: MatrixValidationScope = MatrixValidationScope.CONTAINER_USER_SPACE,
    rocm: str = "7.1.0",
    pytorch: str = "rocm7.1",
    image_tag: str = "7.1.0-complete",
    gfx: str = "gfx1200",
):
    return MatrixTarget(
        target_id=target_id,
        requested_rocm_user_space_version=rocm,
        docker_image_repository=(
            "rocm/dev-ubuntu-24.04"
            if scope is MatrixValidationScope.CONTAINER_USER_SPACE
            else None
        ),
        docker_image_tag=(
            image_tag if scope is MatrixValidationScope.CONTAINER_USER_SPACE else None
        ),
        pytorch_rocm_target=pytorch,
        validation_scope=scope,
        intended_gpu_architecture=gfx,
    )


def _observed(
    *,
    rocm: str = "7.1.0",
    pytorch: str = "rocm7.1",
    image_tag: str = "7.1.0-complete",
    image_digest: str | None = "sha256:old",
    gfx: str = "gfx1200",
    host: bool = True,
    container: bool = True,
):
    return MatrixObservedEvidence(
        host=MatrixHostEvidence(
            rocm_version=rocm,
            driver_version="6.14.0",
            device_nodes=["/dev/kfd", "/dev/dri/renderD128"],
            source="fixture",
        )
        if host
        else None,
        container=MatrixContainerEvidence(
            rocm_user_space_version=rocm,
            image_repository="rocm/dev-ubuntu-24.04",
            image_tag=image_tag,
            image_digest=image_digest,
        )
        if container
        else None,
        python_dependency=MatrixPythonDependencyEvidence(
            python_version="3.12.10",
            torch_distribution_version="2.7.1",
            torch_version=f"2.7.1+{pytorch}",
            torch_local_version=pytorch,
            torch_rocm_target=pytorch,
            torch_hip_version=rocm,
            torch_device_available=True,
            torchvision_distribution_version="0.22.1",
            triton_rocm_distribution_version="3.3.1",
            triton_rocm_status="installed",
        ),
        dependency_policy=MatrixDependencyPolicyEvidence(
            policy_id=f"torch-{pytorch}",
            expected_local_version=pytorch,
            uv_index_name="pytorch-rocm",
            uv_index_url=f"https://download.pytorch.org/whl/{pytorch}",
            lock_strategy="manual",
            suggested_uv_command="uv sync --all-groups",
            triton_rocm_version="3.3.1",
            triton_rocm_index_name="triton-rocm",
            triton_rocm_index_url="https://example.invalid/triton",
        ),
        toolchain=MatrixToolchainEvidence(
            hipcc_version=f"HIP version: {rocm}",
            toolchain_rocm_version=rocm,
            rocm_agent_enumerator_version="1.0.0",
            rocminfo_version="1.0.0",
            tool_statuses={"hipcc": "available"},
        ),
        gpu=MatrixGpuEvidence(
            device_count=1,
            device_name="AMD Radeon RX 9070 XT",
            gfx_architecture=gfx,
            visible_device_environment={"HIP_VISIBLE_DEVICES": "0"},
        ),
    )


def _entry(
    target_id: str,
    *,
    status: MatrixCompatibilityStatus = MatrixCompatibilityStatus.CONTAINER_VALIDATED,
    reason_code: MatrixCompatibilityReasonCode = (
        MatrixCompatibilityReasonCode.CONTAINER_USER_SPACE_VALIDATED
    ),
    reason: str = "Container ROCm user-space matched the requested Target.",
    scope: MatrixValidationScope = MatrixValidationScope.CONTAINER_USER_SPACE,
    target_rocm: str = "7.1.0",
    observed_rocm: str = "7.1.0",
    target_pytorch: str = "rocm7.1",
    observed_pytorch: str = "rocm7.1",
    target_image_tag: str = "7.1.0-complete",
    observed_image_tag: str = "7.1.0-complete",
    image_digest: str | None = "sha256:old",
    target_gfx: str = "gfx1200",
    observed_gfx: str = "gfx1200",
    artifacts: list[MatrixArtifactReference] | None = None,
):
    native = scope is MatrixValidationScope.NATIVE_HOST
    return build_matrix_entry(
        target=_target(
            target_id,
            scope=scope,
            rocm=target_rocm,
            pytorch=target_pytorch,
            image_tag=target_image_tag,
            gfx=target_gfx,
        ),
        observed=_observed(
            rocm=observed_rocm,
            pytorch=observed_pytorch,
            image_tag=observed_image_tag,
            image_digest=image_digest,
            gfx=observed_gfx,
            container=not native,
        ),
        status=status,
        reason_code=reason_code,
        reason=reason,
        claim_boundary=MatrixClaimBoundary(
            container_user_space_validated=(
                status is MatrixCompatibilityStatus.CONTAINER_VALIDATED
            ),
            native_host_validated=native
            and status is MatrixCompatibilityStatus.HOST_VALIDATED,
            hardware_validated=status
            in {
                MatrixCompatibilityStatus.CONTAINER_VALIDATED,
                MatrixCompatibilityStatus.HOST_VALIDATED,
            },
        ),
        artifacts=artifacts or [],
    )


def _report(entries, *, generated_at: str = "2026-05-31T09:00:00Z"):
    counts: dict[MatrixCompatibilityStatus, int] = {}
    for entry in entries:
        counts[entry.status] = counts.get(entry.status, 0) + 1
    return RocmCompatibilityMatrixReport(
        generated_at=generated_at,
        entries=list(entries),
        status_counts=counts,
    )


def test_matrix_diff_matches_keys_and_buckets_are_sorted():
    unchanged = _entry("target-a")
    removed = _entry("target-b")
    old = _report([removed, unchanged])
    changed = _entry(
        "target-a",
        status=MatrixCompatibilityStatus.MIXED_VERSION,
        reason_code=MatrixCompatibilityReasonCode.TARGET_OBSERVED_MISMATCH,
        reason="Observed PyTorch ROCm target drifted.",
        observed_pytorch="rocm7.0",
    )
    added = _entry("target-c")
    new = _report([added, changed])

    diff = diff_matrix_reports(old, new, old_label="old", new_label="new")
    payload = diff.to_dict()

    assert payload["schema_version"] == "sol_execbench.rocm_compatibility_matrix_diff.v1"
    assert payload["old_report"]["label"] == "old"
    assert payload["new_report"]["label"] == "new"
    assert payload["summary_counts"] == {
        "added": 1,
        "changed": 1,
        "removed": 1,
        "unchanged": 0,
    }
    assert [entry["diff_key"] for entry in payload["entry_diffs"]] == [
        "target-a|container_user_space",
        "target-b|container_user_space",
        "target-c|container_user_space",
    ]
    assert [entry["kind"] for entry in payload["entry_diffs"]] == [
        "changed",
        "removed",
        "added",
    ]


def test_matrix_diff_reports_required_semantic_field_groups():
    old = _report(
        [
            _entry(
                "target-a",
                artifacts=[
                    MatrixArtifactReference(
                        artifact_id="probe",
                        kind="probe_json",
                        path="artifacts/old.json",
                    )
                ],
            )
        ]
    )
    new = _report(
        [
            _entry(
                "target-a",
                status=MatrixCompatibilityStatus.MIXED_VERSION,
                reason_code=MatrixCompatibilityReasonCode.TARGET_OBSERVED_MISMATCH,
                reason="Observed ROCm stack drifted.",
                target_rocm="7.1.1",
                observed_rocm="7.0.0",
                target_pytorch="rocm7.1",
                observed_pytorch="rocm7.0",
                target_image_tag="7.1.1-complete",
                observed_image_tag="7.0.0-complete",
                image_digest="sha256:new",
                target_gfx="gfx1201",
                observed_gfx="gfx1201",
                artifacts=[
                    MatrixArtifactReference(
                        artifact_id="probe",
                        kind="probe_json",
                        path="artifacts/new.json",
                    )
                ],
            )
        ]
    )

    changed = diff_matrix_reports(old, new).to_dict()["entry_diffs"][0]

    assert changed["kind"] == "changed"
    assert changed["severity"] == MatrixDiffSeverity.VALIDATION_DOWNGRADE.value
    assert set(changed["semantic_changes"]) == {
        "artifacts",
        "claim_boundary",
        "observed.container",
        "observed.dependency_policy",
        "observed.gpu",
        "observed.host",
        "observed.python_dependency",
        "observed.toolchain",
        "reason_code",
        "status",
        "target",
    }


def test_matrix_diff_normalizes_input_and_artifact_ordering():
    artifact_a = MatrixArtifactReference(
        artifact_id="a",
        kind="probe_json",
        path="artifacts/a.json",
    )
    artifact_b = MatrixArtifactReference(
        artifact_id="b",
        kind="probe_json",
        path="artifacts/b.json",
    )
    old = _report([_entry("target-a", artifacts=[artifact_b, artifact_a])])
    new = _report([_entry("target-a", artifacts=[artifact_a, artifact_b])])

    diff = diff_matrix_reports(old, new).to_dict()

    assert diff["summary_counts"]["unchanged"] == 1
    assert diff["entry_diffs"][0]["kind"] == "unchanged"
    assert diff["entry_diffs"][0]["semantic_changes"] == {}
    assert json.dumps(diff, sort_keys=True) == json.dumps(
        diff_matrix_reports(old, new).to_dict(),
        sort_keys=True,
    )


def test_matrix_diff_reports_clock_evidence_metadata_without_entry_churn():
    entry = _entry("target-a")
    old = _report([entry], generated_at="2026-05-31T09:00:00Z")
    new = _report([entry], generated_at="2026-05-31T10:00:00Z")

    diff = diff_matrix_reports(old, new)
    payload = diff.to_dict()
    markdown = matrix_report_diff_to_markdown(diff)

    assert payload["summary_counts"] == {
        "added": 0,
        "changed": 0,
        "removed": 0,
        "unchanged": 1,
    }
    assert payload["entry_diffs"][0]["kind"] == "unchanged"
    assert payload["entry_diffs"][0]["semantic_changes"] == {}
    assert payload["report_semantic_changes"] == {
        "clock_evidence_metadata": {
            "old": {"generated_at": "2026-05-31T09:00:00Z"},
            "new": {"generated_at": "2026-05-31T10:00:00Z"},
        }
    }
    assert "clock_evidence_metadata" in markdown
    assert "2026-05-31T09:00:00Z" in markdown
    assert "2026-05-31T10:00:00Z" in markdown
    assert json.dumps(payload, sort_keys=True) == json.dumps(
        diff_matrix_reports(old, new).to_dict(),
        sort_keys=True,
    )


def test_matrix_diff_rejects_duplicate_diff_keys():
    duplicate = _entry("target-a")
    report = _report([duplicate, duplicate])

    with pytest.raises(ValueError, match="Duplicate Matrix diff key"):
        diff_matrix_reports(report, _report([]))


def test_matrix_diff_markdown_is_severity_ranked_and_diagnostic_only():
    old = _report([_entry("target-a")])
    new = _report(
        [
            _entry(
                "target-a",
                status=MatrixCompatibilityStatus.RUNTIME_UNAVAILABLE,
                reason_code=MatrixCompatibilityReasonCode.ROCM_RUNTIME_UNAVAILABLE,
                reason="Required ROCm runtime devices were unavailable.",
            )
        ]
    )

    diff = diff_matrix_reports(old, new)
    markdown = matrix_report_diff_to_markdown(diff)

    assert "Diagnostic-only ROCm Compatibility Matrix diff" in markdown
    assert "Docker/container evidence does not imply native-host validation" in markdown
    assert "score authority" in markdown
    assert "paper-parity authority" in markdown
    assert "leaderboard authority" in markdown
    assert "runtime_unavailability" in markdown


def test_load_matrix_report_validates_payload_and_rejects_authority_escalation(tmp_path):
    payload = _report([_entry("target-a")]).model_dump(mode="json")
    path = tmp_path / "matrix.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert load_matrix_report(path).schema_version == (
        ROCM_COMPATIBILITY_MATRIX_SCHEMA_VERSION
    )

    payload["entries"][0]["claim_boundary"]["score_authority"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid Matrix report"):
        load_matrix_report(path)


def test_diff_matrix_reports_script_writes_deterministic_json_and_markdown(
    tmp_path,
    monkeypatch,
):
    module = _load_script()
    old_path = tmp_path / "old.json"
    new_path = tmp_path / "new.json"
    json_out = tmp_path / "diff.json"
    markdown_out = tmp_path / "diff.md"
    old_path.write_text(
        json.dumps(_report([_entry("target-a")]).model_dump(mode="json")),
        encoding="utf-8",
    )
    new_path.write_text(
        json.dumps(
            _report(
                [
                    _entry(
                        "target-a",
                        status=MatrixCompatibilityStatus.MIXED_VERSION,
                        reason_code=(
                            MatrixCompatibilityReasonCode.TARGET_OBSERVED_MISMATCH
                        ),
                        reason="Observed PyTorch ROCm target drifted.",
                        observed_pytorch="rocm7.0",
                    )
                ]
            ).model_dump(mode="json")
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "diff_matrix_reports.py",
            str(old_path),
            str(new_path),
            "--json-out",
            str(json_out),
            "--markdown-out",
            str(markdown_out),
        ],
    )
    assert module.main() == 0
    first_json = json_out.read_text(encoding="utf-8")
    first_markdown = markdown_out.read_text(encoding="utf-8")
    assert module.main() == 0
    assert json_out.read_text(encoding="utf-8") == first_json
    assert markdown_out.read_text(encoding="utf-8") == first_markdown
    assert "mixed_version_drift" in first_json
    assert "Diagnostic-only ROCm Compatibility Matrix diff" in first_markdown


def test_matrix_diff_script_is_not_primary_sol_execbench_cli_option():
    module = _load_script()
    assert "--json-out" in module.build_parser().format_help()

    result = CliRunner().invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "diff_matrix_reports" not in result.output
    assert "--json-out" not in result.output
