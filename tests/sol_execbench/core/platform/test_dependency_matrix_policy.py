from __future__ import annotations

from pathlib import Path

from sol_execbench.core.platform.compatibility import (
    MatrixClaimBoundary,
    MatrixCompatibilityReasonCode,
    MatrixCompatibilityStatus,
    MatrixObservedEvidence,
    build_matrix_entry,
)
from sol_execbench.core.platform.dependency_matrix import (
    dependency_policy_evidence_for_target,
    load_docker_target_dependency_policy,
)
from sol_execbench.core.platform.docker_matrix import (
    load_docker_target_manifest,
    select_docker_target,
    to_matrix_target,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
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
    policy_ids: set[str] = set()

    assert manifest.default_target_id == "rocm-10.1.0-ubuntu-26.04-container"
    assert (
        manifest.targets_by_id[
            manifest.default_target_id
        ].docker_image_repository
        == "rocm/pytorch"
    )
    assert manifest.targets_by_id[
        manifest.default_target_id
    ].docker_image_tag == (
        "rocm10.1.0_ubuntu26.04_py3.14_pytorch_release_2.14.0"
    )
    for target in manifest.targets:
        policy = load_docker_target_dependency_policy(target)
        assert set(policy.model_dump(mode="json")) == EXPECTED_POLICY_FIELDS
        assert policy.policy_id not in policy_ids
        policy_ids.add(policy.policy_id)
        assert policy.policy_id
        assert target.pytorch_rocm_target == policy.expected_local_version
        assert target.pytorch_rocm_target is not None
        assert (
            target.pytorch_rocm_target.replace("rocm", "").replace(".", "")
            in policy.uv_index_name
        )
        assert policy.suggested_uv_command
        assert policy.uv_index_url.startswith(
            "https://stable.repo.amd.com/rocm/whl-next/",
        )
        assert (
            policy.triton_rocm_index_url
            == "https://stable.repo.amd.com/rocm/whl-next/"
        )


def test_default_target_policy_matches_rocm_10_1_project_default() -> None:
    selection = select_docker_target(None, manifest_path=MANIFEST_PATH)
    policy = load_docker_target_dependency_policy(selection.target)

    assert selection.target_id == "rocm-10.1.0-ubuntu-26.04-container"
    assert selection.target.docker_image_repository == "rocm/pytorch"
    assert (
        selection.target.docker_image_tag
        == "rocm10.1.0_ubuntu26.04_py3.14_pytorch_release_2.14.0"
    )
    assert policy.torch_version == "2.14.0+rocm10.1.0"
    assert policy.torchvision_version == "0.29.0a0+rocm10.1.0"
    assert policy.expected_local_version == "rocm10.1.0"
    assert policy.uv_index_name == "amd-rocm1010"
    assert policy.uv_index_url == "https://stable.repo.amd.com/rocm/whl-next/"
    assert policy.lock_strategy == "project_default"
    assert policy.triton_rocm_version == "3.8.0+git669b31ac.rocm10.1.0"
    assert policy.triton_rocm_index_name == "amd-rocm1010"
    assert (
        policy.triton_rocm_index_url
        == "https://stable.repo.amd.com/rocm/whl-next/"
    )


def test_manifest_has_one_supported_official_target() -> None:
    manifest = load_docker_target_manifest(MANIFEST_PATH)

    assert len(manifest.targets) == 1
    assert manifest.targets[0].target_id == manifest.default_target_id


def test_default_project_dependency_path_matches_rocm_10_1() -> None:
    pyproject = PYPROJECT_PATH.read_text()
    uv_lock = UV_LOCK_PATH.read_text()

    assert "torch[device-gfx1200]==2.14.0+rocm10.1.0" in pyproject
    assert "torchvision[device-gfx1200]==0.29.0a0+rocm10.1.0" in pyproject
    assert "triton==3.8.0+git669b31ac.rocm10.1.0" in pyproject
    assert "https://stable.repo.amd.com/rocm/whl-next/" in pyproject
    assert "amd-rocm1010" in pyproject
    assert "https://stable.repo.amd.com/rocm/whl-next/" in uv_lock


def test_default_docker_policy_matches_project_dependency_files() -> None:
    selection = select_docker_target(None, manifest_path=MANIFEST_PATH)
    policy = load_docker_target_dependency_policy(selection.target)
    pyproject = PYPROJECT_PATH.read_text()
    uv_lock = UV_LOCK_PATH.read_text()

    assert policy.lock_strategy == "project_default"
    for expected in (
        f"torch[device-gfx1200]=={policy.torch_version}",
        f"torchvision[device-gfx1200]=={policy.torchvision_version}",
        f"triton=={policy.triton_rocm_version}",
        policy.uv_index_name,
        policy.uv_index_url,
        policy.triton_rocm_index_name,
        policy.triton_rocm_index_url,
    ):
        assert expected in pyproject

    for expected in (
        policy.torch_version,
        policy.torchvision_version,
        policy.uv_index_url,
        policy.triton_rocm_index_url,
    ):
        assert expected in uv_lock


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
        "expected_local_version": "rocm10.1.0",
        "uv_index_name": "amd-rocm1010",
        "uv_index_url": "https://stable.repo.amd.com/rocm/whl-next/",
        "lock_strategy": "project_default",
        "suggested_uv_command": policy.suggested_uv_command,
        "triton_rocm_version": "3.8.0+git669b31ac.rocm10.1.0",
        "triton_rocm_index_name": "amd-rocm1010",
        "triton_rocm_index_url": "https://stable.repo.amd.com/rocm/whl-next/",
    }
