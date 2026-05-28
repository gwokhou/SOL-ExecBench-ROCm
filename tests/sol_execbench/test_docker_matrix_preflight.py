from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from sol_execbench.core.compatibility import (
    MatrixCompatibilityReasonCode,
    MatrixCompatibilityStatus,
)
from sol_execbench.core.docker_matrix import (
    DockerPreflightObservation,
    classify_docker_preflight,
    docker_build_args_for_target,
    select_docker_target,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "docker" / "rocm-targets.json"


def _default_observation(**overrides: object) -> DockerPreflightObservation:
    selection = select_docker_target(None, manifest_path=MANIFEST_PATH)
    values = {
        "docker_context": "default",
        "docker_host": "unix:///var/run/docker.sock",
        "dev_kfd_present": True,
        "dev_kfd_accessible": True,
        "dev_dri_present": True,
        "dev_dri_accessible": True,
        "gpu_accessible": True,
        "selected_target": selection.target,
        "image_repository": selection.target.docker_image_repository,
        "image_tag": selection.target.docker_image_tag,
        "image_digest": None,
        "build_args": docker_build_args_for_target(selection.target),
    }
    values.update(overrides)
    return DockerPreflightObservation(**values)


@pytest.mark.parametrize(
    ("observation", "reason_fragment"),
    [
        (
            _default_observation(
                docker_context="desktop-linux",
                docker_host="unix:///home/user/.docker/desktop/docker.sock",
            ),
            "Docker Desktop",
        ),
        (_default_observation(dev_kfd_present=False), "/dev/kfd"),
        (_default_observation(dev_dri_present=False), "/dev/dri"),
        (_default_observation(gpu_accessible=False), "GPU access"),
    ],
)
def test_preflight_unavailable_cases_block_benchmark_before_execution(
    observation: DockerPreflightObservation,
    reason_fragment: str,
) -> None:
    result = classify_docker_preflight(observation)

    assert result.entry.status is MatrixCompatibilityStatus.RUNTIME_UNAVAILABLE
    assert (
        result.entry.reason_code
        is MatrixCompatibilityReasonCode.ROCM_RUNTIME_UNAVAILABLE
    )
    assert reason_fragment in result.entry.reason
    assert result.decision.benchmark_allowed is False
    assert result.decision.probes_allowed is True
    assert result.decision.smoke_allowed is False
    assert result.decision.container_user_space_validated is False
    assert result.decision.native_host_validated is False


def test_preflight_records_requested_image_and_nullable_digest() -> None:
    result = classify_docker_preflight(_default_observation(dev_dri_present=False))

    assert result.entry.observed.container is not None
    assert result.entry.observed.container.image_repository == (
        "rocm/dev-ubuntu-24.04"
    )
    assert result.entry.observed.container.image_tag == "7.1.1-complete"
    assert result.entry.observed.container.image_digest is None


def test_preflight_records_device_nodes_and_visible_gpu_environment() -> None:
    result = classify_docker_preflight(
        _default_observation(
            dev_kfd_present=False,
            visible_device_environment={"HIP_VISIBLE_DEVICES": "0"},
        )
    )

    assert result.entry.observed.host is not None
    assert result.entry.observed.host.device_nodes == ["/dev/dri"]
    assert result.entry.observed.host.source == "docker_preflight"
    assert result.entry.observed.gpu is not None
    assert result.entry.observed.gpu.visible_device_environment == {
        "HIP_VISIBLE_DEVICES": "0"
    }


def test_preflight_result_payload_contains_build_args_and_decision_flags() -> None:
    result = classify_docker_preflight(_default_observation(gpu_accessible=False))
    payload = result.to_preview_payload()

    assert payload["target_id"]
    assert payload["image_repository"] == "rocm/dev-ubuntu-24.04"
    assert payload["image_tag"] == "7.1.1-complete"
    assert payload["image_digest"] is None
    assert payload["build_args"]["ROCM_DOCKER_IMAGE"] == "rocm/dev-ubuntu-24.04"
    assert payload["build_args"]["ROCM_DOCKER_TAG"] == "7.1.1-complete"
    assert payload["status"] == "runtime_unavailable"
    assert payload["reason_code"] == "rocm_runtime_unavailable"
    assert payload["benchmark_allowed"] is False
    assert payload["score_authority"] is False
    assert payload["paper_parity_authority"] is False
    assert payload["leaderboard_authority"] is False


def test_module_main_emits_preflight_json_from_explicit_observations() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "sol_execbench.core.docker_matrix",
            "preflight",
            "--manifest",
            str(MANIFEST_PATH),
            "--docker-context",
            "desktop-linux",
            "--docker-host",
            "unix:///home/user/.docker/desktop/docker.sock",
            "--dev-kfd-present",
            "true",
            "--dev-kfd-accessible",
            "true",
            "--dev-dri-present",
            "true",
            "--dev-dri-accessible",
            "true",
            "--gpu-accessible",
            "false",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)

    assert payload["target_id"]
    assert payload["validation_scope"] == "container_user_space"
    assert payload["image_repository"] == "rocm/dev-ubuntu-24.04"
    assert payload["image_tag"] == "7.1.1-complete"
    assert payload["image_digest"] is None
    assert payload["build_args"]["ROCM_DOCKER_IMAGE"] == "rocm/dev-ubuntu-24.04"
    assert payload["build_args"]["ROCM_DOCKER_TAG"] == "7.1.1-complete"
    assert payload["status"] == "runtime_unavailable"
    assert payload["reason_code"] == "rocm_runtime_unavailable"
    assert payload["benchmark_allowed"] is False
    assert payload["score_authority"] is False
    assert payload["paper_parity_authority"] is False
    assert payload["leaderboard_authority"] is False


def test_module_main_rejects_invalid_preflight_boolean_without_traceback() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "sol_execbench.core.docker_matrix",
            "preflight",
            "--manifest",
            str(MANIFEST_PATH),
            "--docker-context",
            "default",
            "--docker-host",
            "unix:///var/run/docker.sock",
            "--dev-kfd-present",
            "true",
            "--dev-kfd-accessible",
            "true",
            "--dev-dri-present",
            "true",
            "--dev-dri-accessible",
            "true",
            "--gpu-accessible",
            "maybe",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "expected boolean value" in completed.stderr
    assert "Traceback" not in completed.stderr
