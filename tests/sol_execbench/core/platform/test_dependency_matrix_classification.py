from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from sol_execbench.core.platform.compatibility import (
    MatrixCompatibilityReasonCode,
    MatrixCompatibilityStatus,
)
from sol_execbench.core.platform.dependency_matrix import (
    PytorchDependencyObservation,
    PytorchDependencyPolicy,
    classify_dependency_preflight,
    load_docker_target_dependency_policy,
)
from sol_execbench.core.platform.docker_matrix import (
    load_docker_target_manifest,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
MANIFEST_PATH = REPO_ROOT / "docker" / "rocm-targets.json"


def _target(target_id: str = "rocm-10.1.0-ubuntu-26.04-container"):
    manifest = load_docker_target_manifest(MANIFEST_PATH)
    return manifest.targets_by_id[target_id]


def _default_policy() -> PytorchDependencyPolicy:
    return load_docker_target_dependency_policy(_target())


def _matching_observation(**overrides: object) -> PytorchDependencyObservation:
    values: dict[str, Any] = {
        "torch_distribution_version": "2.14.0+rocm10.1.0",
        "torch_version": "2.14.0+rocm10.1.0",
        "torch_local_version": "rocm10.1.0",
        "torch_rocm_target": "rocm10.1.0",
        "torch_hip_version": "7.16.0",
        "torch_cuda_version": None,
        "torch_device_available": True,
        "torch_import_error": None,
        "torchvision_distribution_version": "0.29.0a0+rocm10.1.0",
        "triton_rocm_distribution_version": "3.8.0+git669b31ac.rocm10.1.0",
        "triton_rocm_status": "installed",
        "container_rocm_user_space_version": "10.1.0",
        "hipcc_version": "HIP version: 10.1.0",
        "toolchain_rocm_version": "10.1.0",
    }
    values.update(overrides)
    return PytorchDependencyObservation(**values)


def _assert_no_authority(result) -> None:
    assert result.decision.benchmark_allowed is False
    assert result.decision.container_user_space_validated is False
    assert result.decision.native_host_validated is False
    assert result.decision.score_authority is False
    assert result.decision.paper_parity_authority is False
    assert result.decision.leaderboard_authority is False


def _assert_policy_payload(result) -> None:
    payload = result.entry.model_dump(mode="json")["observed"][
        "dependency_policy"
    ]
    assert payload["policy_id"]
    assert payload["expected_local_version"]
    assert payload["uv_index_name"]
    assert payload["uv_index_url"].startswith(
        "https://stable.repo.amd.com/rocm/whl-next/",
    )
    assert payload["lock_strategy"]
    assert payload["suggested_uv_command"]
    assert payload["triton_rocm_version"]
    assert payload["triton_rocm_index_name"]
    assert (
        payload["triton_rocm_index_url"]
        == "https://stable.repo.amd.com/rocm/whl-next/"
    )


def test_unsupported_policy_is_pytorch_wheel_unavailable() -> None:
    policy = _default_policy().model_copy(
        update={"wheel_availability": "unavailable"},
    )

    result = classify_dependency_preflight(
        target=_target(),
        policy=policy,
        observation=_matching_observation(torch_distribution_version=None),
    )

    assert (
        result.entry.status
        is MatrixCompatibilityStatus.PYTORCH_WHEEL_UNAVAILABLE
    )
    assert (
        result.entry.reason_code
        is MatrixCompatibilityReasonCode.PYTORCH_ROCM_WHEEL_UNAVAILABLE
    )
    assert "unavailable" in result.entry.reason
    assert result.decision.benchmark_allowed is False
    _assert_no_authority(result)
    _assert_policy_payload(result)


def test_torch_import_error_blocks_even_when_metadata_matches_policy() -> None:
    result = classify_dependency_preflight(
        target=_target(),
        policy=_default_policy(),
        observation=_matching_observation(
            torch_import_error="libamdhip64.so: cannot open shared object file",
        ),
    )

    assert (
        result.entry.status
        is MatrixCompatibilityStatus.PYTORCH_WHEEL_UNAVAILABLE
    )
    assert (
        result.entry.reason_code
        is MatrixCompatibilityReasonCode.PYTORCH_ROCM_WHEEL_UNAVAILABLE
    )
    assert "could not be imported" in result.entry.reason
    assert "Dependency stack matches" not in result.entry.reason
    _assert_no_authority(result)
    _assert_policy_payload(result)


@pytest.mark.parametrize(
    ("overrides", "reason_fragment"),
    [
        ({"torch_local_version": None}, "local-version"),
        ({"torch_cuda_version": "12.8"}, "CUDA"),
        (
            {"torch_local_version": "rocm7.0", "torch_rocm_target": "rocm7.0"},
            "rocm",
        ),
        ({"torch_hip_version": "7.0.0"}, "HIP"),
        ({"torchvision_distribution_version": "0.25.0+rocm7.0"}, "torchvision"),
        ({"triton_rocm_distribution_version": None}, "ROCm Triton"),
        ({"triton_rocm_status": "missing"}, "ROCm Triton"),
        ({"container_rocm_user_space_version": "7.0.2"}, "container"),
        ({"toolchain_rocm_version": "7.0.2"}, "toolchain"),
    ],
)
def test_dependency_mismatches_classify_as_mixed_version(
    overrides: dict[str, object],
    reason_fragment: str,
) -> None:
    result = classify_dependency_preflight(
        target=_target(),
        policy=_default_policy(),
        observation=_matching_observation(**overrides),
    )

    assert result.entry.status is MatrixCompatibilityStatus.MIXED_VERSION
    assert (
        result.entry.reason_code
        is MatrixCompatibilityReasonCode.TARGET_OBSERVED_MISMATCH
    )
    assert reason_fragment in result.entry.reason
    assert result.decision.benchmark_allowed is False
    _assert_no_authority(result)
    _assert_policy_payload(result)


def test_matching_default_rocm_10_1_dependency_stack_is_not_tested() -> None:
    result = classify_dependency_preflight(
        target=_target(),
        policy=_default_policy(),
        observation=_matching_observation(),
    )

    assert result.entry.status is MatrixCompatibilityStatus.NOT_TESTED
    assert (
        result.entry.reason_code
        is MatrixCompatibilityReasonCode.TARGET_NOT_TESTED
    )
    assert result.decision.benchmark_allowed is False
    _assert_no_authority(result)
    _assert_policy_payload(result)


def test_mixed_version_debug_override_allows_probe_or_smoke_without_clean_claims() -> (
    None
):
    result = classify_dependency_preflight(
        target=_target(),
        policy=_default_policy(),
        observation=_matching_observation(torch_local_version="rocm7.0"),
        allow_mixed_version_debug=True,
    )

    assert result.entry.status is MatrixCompatibilityStatus.MIXED_VERSION
    assert result.decision.probes_allowed is True
    assert result.decision.smoke_allowed is True
    assert result.decision.benchmark_allowed is False
    assert result.decision.container_user_space_validated is False
    assert result.decision.native_host_validated is False
    assert result.decision.score_authority is False
    assert result.decision.paper_parity_authority is False
    assert result.decision.leaderboard_authority is False
    _assert_policy_payload(result)
