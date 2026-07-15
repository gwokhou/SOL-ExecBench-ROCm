from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from sol_execbench.core.platform.compatibility import (
    MatrixCompatibilityReasonCode,
    MatrixCompatibilityStatus,
    MatrixValidationScope,
)
from sol_execbench.core.platform.docker_matrix import (
    docker_build_args_for_target,
    load_docker_target_manifest,
    preview_docker_target_selection,
    select_docker_target,
    to_matrix_target,
)


REPO_ROOT = Path(__file__).resolve().parents[4]
MANIFEST_PATH = REPO_ROOT / "docker" / "rocm-targets.json"


def test_manifest_declares_default_and_configured_rocm_complete_targets() -> None:
    manifest = load_docker_target_manifest(MANIFEST_PATH)

    assert manifest.default_target_id
    assert manifest.targets_by_id[
        manifest.default_target_id
    ].docker_image_repository == ("rocm/dev-ubuntu-24.04")
    assert manifest.targets_by_id[manifest.default_target_id].docker_image_tag == (
        "7.2-complete"
    )
    tags = {target.docker_image_tag for target in manifest.targets}
    assert any(tag.startswith("7.0.") and tag.endswith("-complete") for tag in tags)
    assert any(tag.startswith("7.2") and tag.endswith("-complete") for tag in tags)
    assert {target.validation_scope for target in manifest.targets} == {
        MatrixValidationScope.CONTAINER_USER_SPACE
    }

    raw = MANIFEST_PATH.read_text()
    assert "rocm/dev-ubuntu-24.04" in raw
    assert "7.2-complete" in raw


def test_default_selection_uses_rocm_7_2_docker_image() -> None:
    selection = select_docker_target(None, manifest_path=MANIFEST_PATH)

    assert selection.target_id == selection.target.target_id
    assert selection.target.docker_image_repository == "rocm/dev-ubuntu-24.04"
    assert selection.target.docker_image_tag == "7.2-complete"
    assert selection.status is None
    assert selection.entry is None
    assert selection.decision is None


def test_selection_converts_declared_target_to_phase_78_matrix_target() -> None:
    selection = select_docker_target(None, manifest_path=MANIFEST_PATH)
    matrix_target = to_matrix_target(selection.target)

    assert matrix_target.target_id == selection.target.target_id
    assert matrix_target.requested_rocm_user_space_version.startswith("7.2.")
    assert matrix_target.docker_image_repository == "rocm/dev-ubuntu-24.04"
    assert matrix_target.docker_image_tag == "7.2-complete"
    assert matrix_target.pytorch_rocm_target == "rocm7.2"
    assert matrix_target.validation_scope is MatrixValidationScope.CONTAINER_USER_SPACE


def test_docker_build_args_use_requested_manifest_repository_and_tag() -> None:
    selection = select_docker_target(None, manifest_path=MANIFEST_PATH)

    assert docker_build_args_for_target(selection.target) == {
        "ROCM_DOCKER_IMAGE": "rocm/dev-ubuntu-24.04",
        "ROCM_DOCKER_TAG": "7.2-complete",
        "PYTORCH_TORCH_VERSION": "2.11.0+rocm7.2",
        "PYTORCH_TORCHVISION_VERSION": "0.26.0+rocm7.2",
        "PYTORCH_ROCM_INDEX_URL": "https://download.pytorch.org/whl/rocm7.2",
        "TRITON_ROCM_VERSION": "3.6.0",
        "TRITON_ROCM_INDEX_URL": "https://download.pytorch.org/whl/",
    }


def test_unknown_target_rejected_without_explicit_override() -> None:
    with pytest.raises(ValueError, match="Unknown Docker Target"):
        select_docker_target("rocm-9.9-unknown", manifest_path=MANIFEST_PATH)


def test_unknown_target_override_is_not_tested_and_non_authoritative() -> None:
    selection = select_docker_target(
        "rocm-9.9-unknown",
        manifest_path=MANIFEST_PATH,
        allow_unknown_override=True,
        override_image_repository="example.invalid/rocm",
        override_image_tag="9.9-complete",
    )

    assert selection.entry is not None
    assert selection.decision is not None
    assert selection.status is MatrixCompatibilityStatus.NOT_TESTED
    assert (
        selection.entry.reason_code is MatrixCompatibilityReasonCode.TARGET_NOT_TESTED
    )
    assert selection.decision.benchmark_allowed is False
    assert selection.decision.container_user_space_validated is False
    assert selection.decision.native_host_validated is False
    assert selection.decision.score_authority is False
    assert selection.decision.paper_parity_authority is False
    assert selection.decision.leaderboard_authority is False
    assert selection.entry.claim_boundary.container_user_space_validated is False
    assert selection.entry.claim_boundary.native_host_validated is False
    assert selection.entry.claim_boundary.score_authority is False
    assert selection.entry.claim_boundary.paper_parity_authority is False
    assert selection.entry.claim_boundary.leaderboard_authority is False


def test_default_preview_json_is_shell_consumable_without_docker() -> None:
    payload = preview_docker_target_selection(manifest_path=MANIFEST_PATH)

    assert payload["target_id"]
    assert payload["image_repository"] == "rocm/dev-ubuntu-24.04"
    assert payload["image_tag"] == "7.2-complete"
    assert payload["image_digest"] is None
    assert payload["validation_scope"] == "container_user_space"
    assert payload["build_args"]["ROCM_DOCKER_IMAGE"] == "rocm/dev-ubuntu-24.04"
    assert payload["build_args"]["ROCM_DOCKER_TAG"] == "7.2-complete"
    assert payload["build_args"]["PYTORCH_TORCH_VERSION"] == "2.11.0+rocm7.2"
    assert payload["build_args"]["PYTORCH_TORCHVISION_VERSION"] == ("0.26.0+rocm7.2")
    assert payload["build_args"]["PYTORCH_ROCM_INDEX_URL"] == (
        "https://download.pytorch.org/whl/rocm7.2"
    )
    assert payload["build_args"]["TRITON_ROCM_VERSION"] == "3.6.0"
    assert payload["build_args"]["TRITON_ROCM_INDEX_URL"] == (
        "https://download.pytorch.org/whl/"
    )
    assert payload["status"] == "not_tested"
    assert payload["reason_code"] == "target_not_tested"
    assert payload["benchmark_allowed"] is False
    assert payload["score_authority"] is False
    assert payload["paper_parity_authority"] is False
    assert payload["leaderboard_authority"] is False


def test_non_default_preview_json_selects_declared_target() -> None:
    manifest = load_docker_target_manifest(MANIFEST_PATH)
    non_default = next(
        target
        for target in manifest.targets
        if target.target_id != manifest.default_target_id
    )

    payload = preview_docker_target_selection(
        target_id=non_default.target_id,
        manifest_path=MANIFEST_PATH,
    )

    assert payload["target_id"] == non_default.target_id
    assert payload["image_repository"] == non_default.docker_image_repository
    assert payload["image_tag"] == non_default.docker_image_tag
    assert payload["build_args"]["ROCM_DOCKER_TAG"] == non_default.docker_image_tag
    assert non_default.pytorch_dependency_policy is not None
    assert (
        payload["build_args"]["PYTORCH_TORCH_VERSION"]
        == (non_default.pytorch_dependency_policy["torch_version"])
    )
    assert (
        payload["build_args"]["PYTORCH_ROCM_INDEX_URL"]
        == (non_default.pytorch_dependency_policy["uv_index_url"])
    )


def test_unknown_preview_rejected_without_override() -> None:
    with pytest.raises(ValueError, match="Unknown Docker Target"):
        preview_docker_target_selection(
            target_id="rocm-9.9-unknown",
            manifest_path=MANIFEST_PATH,
        )


def test_unknown_override_preview_is_diagnostic_not_tested_only() -> None:
    payload = preview_docker_target_selection(
        target_id="rocm-9.9-unknown",
        manifest_path=MANIFEST_PATH,
        allow_unknown_override=True,
        override_image_repository="example.invalid/rocm",
        override_image_tag="9.9-complete",
    )

    assert payload["target_id"] == "unsafe-untested-rocm-9.9-unknown"
    assert payload["image_repository"] == "example.invalid/rocm"
    assert payload["image_tag"] == "9.9-complete"
    assert payload["status"] == "not_tested"
    assert payload["reason_code"] == "target_not_tested"
    assert payload["benchmark_allowed"] is False
    assert payload["container_user_space_validated"] is False
    assert payload["native_host_validated"] is False
    assert payload["score_authority"] is False
    assert payload["paper_parity_authority"] is False
    assert payload["leaderboard_authority"] is False


def test_module_main_emits_default_preview_json_without_live_docker() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "sol_execbench.core.platform.docker_matrix",
            "preview",
            "--manifest",
            str(MANIFEST_PATH),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)

    assert payload["build_args"]["ROCM_DOCKER_IMAGE"] == "rocm/dev-ubuntu-24.04"
    assert payload["build_args"]["ROCM_DOCKER_TAG"] == "7.2-complete"
