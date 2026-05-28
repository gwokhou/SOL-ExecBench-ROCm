from __future__ import annotations

from pathlib import Path

from sol_execbench.core.dependency_matrix import (
    dependency_policy_evidence_for_target,
    load_docker_target_dependency_policy,
)
from sol_execbench.core.docker_matrix import (
    load_docker_target_manifest,
    select_docker_target,
    to_matrix_target,
)
from sol_execbench.core.compatibility import (
    MatrixClaimBoundary,
    MatrixCompatibilityReasonCode,
    MatrixCompatibilityStatus,
    MatrixObservedEvidence,
    build_matrix_entry,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "docker" / "rocm-targets.json"
PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"
UV_LOCK_PATH = REPO_ROOT / "uv.lock"


EXPECTED_POLICY_FIELDS = {
    "policy_id",
    "wheel_availability",
    "torch_version",
    "torchvision_version",
    "expected_local_version",
    "uv_index_name",
    "uv_index_url",
    "lock_strategy",
    "suggested_uv_command",
    "triton_rocm_version",
    "triton_rocm_index_name",
    "triton_rocm_index_url",
}


def test_manifest_records_dependency_policy_for_every_declared_target() -> None:
    manifest = load_docker_target_manifest(MANIFEST_PATH)

    assert manifest.default_target_id == "rocm-7.1.1-ubuntu-24.04-container"
    assert manifest.targets_by_id[
        manifest.default_target_id
    ].docker_image_repository == "rocm/dev-ubuntu-24.04"
    assert manifest.targets_by_id[manifest.default_target_id].docker_image_tag == (
        "7.1.1-complete"
    )
    for target in manifest.targets:
        policy = load_docker_target_dependency_policy(target)
        assert set(policy.model_dump(mode="json")) == EXPECTED_POLICY_FIELDS
        assert policy.policy_id
        assert policy.suggested_uv_command
        assert policy.uv_index_url.startswith("https://download.pytorch.org/whl/")
        assert policy.triton_rocm_index_url == "https://download.pytorch.org/whl/"


def test_default_target_policy_preserves_rocm_7_1_project_default() -> None:
    selection = select_docker_target(None, manifest_path=MANIFEST_PATH)
    policy = load_docker_target_dependency_policy(selection.target)

    assert selection.target_id == "rocm-7.1.1-ubuntu-24.04-container"
    assert selection.target.docker_image_repository == "rocm/dev-ubuntu-24.04"
    assert selection.target.docker_image_tag == "7.1.1-complete"
    assert policy.torch_version == "2.10.0+rocm7.1"
    assert policy.torchvision_version == "0.25.0+rocm7.1"
    assert policy.expected_local_version == "rocm7.1"
    assert policy.uv_index_name == "pytorch-rocm71"
    assert policy.uv_index_url == "https://download.pytorch.org/whl/rocm7.1"
    assert policy.lock_strategy == "project_default"
    assert policy.triton_rocm_version == "3.6.0"
    assert policy.triton_rocm_index_name == "pytorch-rocm-root"
    assert policy.triton_rocm_index_url == "https://download.pytorch.org/whl/"


def test_non_default_target_policies_record_explicit_workflows() -> None:
    manifest = load_docker_target_manifest(MANIFEST_PATH)
    rocm_70 = load_docker_target_dependency_policy(
        manifest.targets_by_id["rocm-7.0.2-ubuntu-24.04-container"]
    )
    rocm_72 = load_docker_target_dependency_policy(
        manifest.targets_by_id["rocm-7.2.0-ubuntu-24.04-container"]
    )

    assert rocm_70.torch_version == "2.10.0+rocm7.0"
    assert rocm_70.torchvision_version == "0.25.0+rocm7.0"
    assert rocm_70.expected_local_version == "rocm7.0"
    assert rocm_70.uv_index_name == "pytorch-rocm70"
    assert rocm_70.uv_index_url == "https://download.pytorch.org/whl/rocm7.0"

    assert rocm_72.torch_version == "2.11.0+rocm7.2"
    assert rocm_72.torchvision_version == "0.26.0+rocm7.2"
    assert rocm_72.expected_local_version == "rocm7.2"
    assert rocm_72.uv_index_name == "pytorch-rocm72"
    assert rocm_72.uv_index_url == "https://download.pytorch.org/whl/rocm7.2"


def test_default_project_dependency_path_remains_rocm_7_1() -> None:
    pyproject = PYPROJECT_PATH.read_text()
    uv_lock = UV_LOCK_PATH.read_text()

    for text in (pyproject, uv_lock):
        assert "torch==2.10.0+rocm7.1" in text
        assert "torchvision==0.25.0+rocm7.1" in text
        assert "triton-rocm==3.6.0" in text
        assert "https://download.pytorch.org/whl/rocm7.1" in text
        assert "https://download.pytorch.org/whl/" in text
    assert "pytorch-rocm71" in pyproject
    assert "pytorch-rocm-root" in pyproject


def test_matrix_entry_payload_records_dependency_policy() -> None:
    selection = select_docker_target(None, manifest_path=MANIFEST_PATH)
    policy = dependency_policy_evidence_for_target(selection.target)
    entry = build_matrix_entry(
        target=to_matrix_target(selection.target),
        observed=MatrixObservedEvidence(dependency_policy=policy),
        status=MatrixCompatibilityStatus.NOT_TESTED,
        reason_code=MatrixCompatibilityReasonCode.TARGET_NOT_TESTED,
        reason="Dependency policy is recorded, but no runtime validation was performed.",
        claim_boundary=MatrixClaimBoundary(
            container_user_space_validated=False,
            native_host_validated=False,
            hardware_validated=False,
        ),
    )
    payload = entry.model_dump(mode="json")

    assert payload["observed"]["dependency_policy"] == {
        "policy_id": policy.policy_id,
        "expected_local_version": "rocm7.1",
        "uv_index_name": "pytorch-rocm71",
        "uv_index_url": "https://download.pytorch.org/whl/rocm7.1",
        "lock_strategy": "project_default",
        "suggested_uv_command": policy.suggested_uv_command,
        "triton_rocm_version": "3.6.0",
        "triton_rocm_index_name": "pytorch-rocm-root",
        "triton_rocm_index_url": "https://download.pytorch.org/whl/",
    }
