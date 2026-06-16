"""PyTorch ROCm dependency policy helpers for declared Docker Targets."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path
from typing import Any
from typing import Literal

from pydantic import ConfigDict

from sol_execbench.core.utils import (
    none_if_requested as _none_if_requested,
    parse_bool as _parse_bool,
)

from sol_execbench.core.compatibility import (
    MatrixClaimBoundary,
    MatrixCompatibilityReasonCode,
    MatrixCompatibilityStatus,
    MatrixContainerEvidence,
    MatrixDependencyPolicyEvidence,
    MatrixEntry,
    MatrixExecutionDecision,
    MatrixObservedEvidence,
    MatrixPythonDependencyEvidence,
    MatrixToolchainEvidence,
    build_matrix_entry,
    classify_matrix_entry_for_execution,
)
from sol_execbench.core.data.base_model import BaseModelWithDocstrings
from sol_execbench.core.docker_matrix import (
    DEFAULT_DOCKER_TARGET_MANIFEST,
    DockerTargetManifestEntry,
    select_docker_target,
    to_matrix_target,
)


_MODEL_CONFIG = ConfigDict(
    extra="forbid",
    frozen=True,
    strict=True,
    use_attribute_docstrings=True,
)


class PytorchDependencyPolicy(BaseModelWithDocstrings):
    """Strict checked-in PyTorch ROCm dependency policy for one Target."""

    model_config = _MODEL_CONFIG

    policy_id: str
    """Stable dependency policy identifier."""
    wheel_availability: Literal["available", "unavailable"]
    """Whether matching PyTorch ROCm wheels are declared available."""
    torch_version: str
    """Expected torch distribution version, including ROCm local tag."""
    torchvision_version: str
    """Expected torchvision distribution version, including ROCm local tag."""
    expected_local_version: str
    """Expected PyTorch ROCm local-version tag."""
    uv_index_name: str
    """uv index name expected to provide torch and torchvision."""
    uv_index_url: str
    """uv index URL expected to provide torch and torchvision."""
    lock_strategy: str
    """Dependency lock or workflow strategy for this policy."""
    suggested_uv_command: str
    """Auditable uv command or workflow users can run for this policy."""
    triton_rocm_version: str
    """Expected triton-rocm distribution version."""
    triton_rocm_index_name: str
    """uv index name expected to provide triton-rocm."""
    triton_rocm_index_url: str
    """uv index URL expected to provide triton-rocm."""


class PytorchDependencyObservation(BaseModelWithDocstrings):
    """Observed installed PyTorch ROCm dependency stack."""

    model_config = _MODEL_CONFIG

    torch_distribution_version: str | None = None
    """Installed torch distribution version."""
    torch_version: str | None = None
    """Observed torch.__version__ value."""
    torch_local_version: str | None = None
    """Observed torch local-version tag, such as rocm7.1."""
    torch_rocm_target: str | None = None
    """Observed PyTorch ROCm target inferred from local-version metadata."""
    torch_hip_version: str | None = None
    """Observed torch.version.hip value."""
    torch_cuda_version: str | None = None
    """Observed torch.version.cuda value."""
    torch_device_available: bool | None = None
    """Whether PyTorch reported device availability."""
    torch_import_error: str | None = None
    """Torch import error when runtime probing failed."""
    torchvision_distribution_version: str | None = None
    """Installed torchvision distribution version."""
    triton_rocm_distribution_version: str | None = None
    """Installed triton-rocm distribution version."""
    triton_rocm_status: str | None = None
    """Observed triton-rocm status, such as installed or missing."""
    container_rocm_user_space_version: str | None = None
    """Observed container ROCm user-space version."""
    hipcc_version: str | None = None
    """Observed hipcc version output."""
    toolchain_rocm_version: str | None = None
    """Observed parsed ROCm toolchain version."""


class DependencyPreflightResult(BaseModelWithDocstrings):
    """Matrix-compatible dependency preflight classification."""

    model_config = _MODEL_CONFIG

    entry: MatrixEntry
    """Diagnostic Matrix Entry produced by dependency classification."""
    decision: MatrixExecutionDecision
    """Pre-benchmark execution decision derived from the Matrix Entry."""
    policy: PytorchDependencyPolicy
    """Selected Target dependency policy used for classification."""

    def to_preview_payload(self) -> dict[str, Any]:
        """Return shell-consumable JSON for dependency preflight classification."""

        entry_payload = self.entry.model_dump(mode="json")
        decision_payload = self.decision.model_dump(mode="json")
        target_payload = entry_payload["target"]
        policy_payload = entry_payload["observed"]["dependency_policy"]
        claim_payload = entry_payload["claim_boundary"]
        return {
            "target_id": target_payload["target_id"],
            "pytorch_rocm_target": target_payload["pytorch_rocm_target"],
            "policy_id": policy_payload["policy_id"],
            "wheel_availability": self.policy.wheel_availability,
            "expected_local_version": policy_payload["expected_local_version"],
            "uv_index_name": policy_payload["uv_index_name"],
            "uv_index_url": policy_payload["uv_index_url"],
            "lock_strategy": policy_payload["lock_strategy"],
            "suggested_uv_command": policy_payload["suggested_uv_command"],
            "torch_version": self.policy.torch_version,
            "torchvision_version": self.policy.torchvision_version,
            "triton_rocm_version": policy_payload["triton_rocm_version"],
            "triton_rocm_index_name": policy_payload["triton_rocm_index_name"],
            "triton_rocm_index_url": policy_payload["triton_rocm_index_url"],
            "status": entry_payload["status"],
            "reason_code": entry_payload["reason_code"],
            "reason": entry_payload["reason"],
            "benchmark_allowed": decision_payload["benchmark_allowed"],
            "probes_allowed": decision_payload["probes_allowed"],
            "smoke_allowed": decision_payload["smoke_allowed"],
            "score_authority": decision_payload["score_authority"],
            "paper_parity_authority": decision_payload["paper_parity_authority"],
            "leaderboard_authority": decision_payload["leaderboard_authority"],
            "container_user_space_validated": decision_payload[
                "container_user_space_validated"
            ],
            "native_host_validated": decision_payload["native_host_validated"],
            "hardware_validated": claim_payload["hardware_validated"],
        }


def load_docker_target_dependency_policy(
    target: DockerTargetManifestEntry,
) -> PytorchDependencyPolicy:
    """Load and strictly validate dependency policy from a Docker Target."""

    if target.pytorch_dependency_policy is None:
        raise ValueError(
            f"Docker Target {target.target_id!r} does not declare "
            "pytorch_dependency_policy."
        )
    return PytorchDependencyPolicy.model_validate(target.pytorch_dependency_policy)


def dependency_policy_evidence_for_target(
    target: DockerTargetManifestEntry,
) -> MatrixDependencyPolicyEvidence:
    """Convert a Docker Target policy into Matrix Entry policy evidence."""

    policy = load_docker_target_dependency_policy(target)
    return MatrixDependencyPolicyEvidence(
        policy_id=policy.policy_id,
        expected_local_version=policy.expected_local_version,
        uv_index_name=policy.uv_index_name,
        uv_index_url=policy.uv_index_url,
        lock_strategy=policy.lock_strategy,
        suggested_uv_command=policy.suggested_uv_command,
        triton_rocm_version=policy.triton_rocm_version,
        triton_rocm_index_name=policy.triton_rocm_index_name,
        triton_rocm_index_url=policy.triton_rocm_index_url,
    )


def collect_pytorch_dependency_observation() -> PytorchDependencyObservation:
    """Collect lightweight dependency observations without requiring ROCm hardware."""

    torch_distribution_version = _distribution_version("torch")
    torchvision_distribution_version = _distribution_version("torchvision")
    triton_rocm_distribution_version = _distribution_version("triton-rocm")
    triton_rocm_status = (
        "installed" if triton_rocm_distribution_version is not None else "missing"
    )
    try:
        import torch
    except ImportError as exc:
        return PytorchDependencyObservation(
            torch_distribution_version=torch_distribution_version,
            torch_local_version=_local_version(torch_distribution_version),
            torch_rocm_target=_local_version(torch_distribution_version),
            torch_import_error=str(exc),
            torchvision_distribution_version=torchvision_distribution_version,
            triton_rocm_distribution_version=triton_rocm_distribution_version,
            triton_rocm_status=triton_rocm_status,
        )

    torch_version = str(getattr(torch, "__version__", ""))
    version = getattr(torch, "version", None)
    hip_version = getattr(version, "hip", None)
    cuda_version = getattr(version, "cuda", None)
    try:
        device_available = bool(torch.cuda.is_available())
    except (RuntimeError, AttributeError):
        device_available = False
    local_version = _local_version(torch_version or torch_distribution_version)
    rocm_version = _collect_rocm_version_file()
    return PytorchDependencyObservation(
        torch_distribution_version=torch_distribution_version,
        torch_version=torch_version,
        torch_local_version=local_version,
        torch_rocm_target=local_version,
        torch_hip_version=hip_version,
        torch_cuda_version=cuda_version,
        torch_device_available=device_available,
        torchvision_distribution_version=torchvision_distribution_version,
        triton_rocm_distribution_version=triton_rocm_distribution_version,
        triton_rocm_status=triton_rocm_status,
        container_rocm_user_space_version=rocm_version,
        hipcc_version=_collect_command_output(["hipcc", "--version"]),
        toolchain_rocm_version=rocm_version,
    )


def _collect_rocm_version_file() -> str | None:
    for path in (
        Path("/opt/rocm/.info/version"),
        Path("/opt/rocm/.info/version-dev"),
    ):
        try:
            version = path.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if version:
            return version
    return None


def _collect_command_output(command: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    output = (completed.stdout or completed.stderr).strip()
    return output or None


def classify_dependency_preflight(
    *,
    target: DockerTargetManifestEntry,
    policy: PytorchDependencyPolicy,
    observation: PytorchDependencyObservation,
    allow_mixed_version_debug: bool = False,
) -> DependencyPreflightResult:
    """Classify a selected Target's observed PyTorch ROCm dependency stack."""

    status, reason_code, reason = _classify_observation(
        target=target,
        policy=policy,
        observation=observation,
    )
    entry = build_matrix_entry(
        target=to_matrix_target(target),
        observed=MatrixObservedEvidence(
            container=MatrixContainerEvidence(
                rocm_user_space_version=observation.container_rocm_user_space_version,
                image_repository=target.docker_image_repository,
                image_tag=target.docker_image_tag,
            ),
            python_dependency=_python_dependency_evidence(observation),
            dependency_policy=dependency_policy_evidence_for_target(target),
            toolchain=MatrixToolchainEvidence(
                hipcc_version=observation.hipcc_version,
                toolchain_rocm_version=observation.toolchain_rocm_version,
            ),
        ),
        status=status,
        reason_code=reason_code,
        reason=reason,
        claim_boundary=MatrixClaimBoundary(
            container_user_space_validated=False,
            native_host_validated=False,
            hardware_validated=False,
        ),
    )
    return DependencyPreflightResult(
        entry=entry,
        decision=classify_matrix_entry_for_execution(
            entry,
            allow_mixed_version_debug=allow_mixed_version_debug,
        ),
        policy=policy,
    )


def _classify_observation(
    *,
    target: DockerTargetManifestEntry,
    policy: PytorchDependencyPolicy,
    observation: PytorchDependencyObservation,
) -> tuple[
    MatrixCompatibilityStatus,
    MatrixCompatibilityReasonCode,
    str,
]:
    if policy.wheel_availability == "unavailable":
        return (
            MatrixCompatibilityStatus.PYTORCH_WHEEL_UNAVAILABLE,
            MatrixCompatibilityReasonCode.PYTORCH_ROCM_WHEEL_UNAVAILABLE,
            f"PyTorch ROCm wheels are declared unavailable by policy {policy.policy_id}.",
        )
    if observation.torch_import_error:
        return (
            MatrixCompatibilityStatus.PYTORCH_WHEEL_UNAVAILABLE,
            MatrixCompatibilityReasonCode.PYTORCH_ROCM_WHEEL_UNAVAILABLE,
            f"Required torch runtime could not be imported: {observation.torch_import_error}",
        )
    if observation.torch_distribution_version is None:
        return (
            MatrixCompatibilityStatus.PYTORCH_WHEEL_UNAVAILABLE,
            MatrixCompatibilityReasonCode.PYTORCH_ROCM_WHEEL_UNAVAILABLE,
            "Required torch distribution is unavailable for this Target policy.",
        )

    mismatch_reason = _mismatch_reason(
        target=target, policy=policy, observation=observation
    )
    if mismatch_reason is not None:
        return (
            MatrixCompatibilityStatus.MIXED_VERSION,
            MatrixCompatibilityReasonCode.TARGET_OBSERVED_MISMATCH,
            mismatch_reason,
        )
    return (
        MatrixCompatibilityStatus.NOT_TESTED,
        MatrixCompatibilityReasonCode.TARGET_NOT_TESTED,
        (
            "Dependency stack matches the selected Target policy, but no benchmark, "
            "container user-space, native host, or hardware validation has been performed."
        ),
    )


def _mismatch_reason(
    *,
    target: DockerTargetManifestEntry,
    policy: PytorchDependencyPolicy,
    observation: PytorchDependencyObservation,
) -> str | None:
    if observation.torch_cuda_version:
        return (
            f"CUDA PyTorch runtime {observation.torch_cuda_version} was observed for "
            f"ROCm Target {target.target_id}."
        )
    if observation.torch_distribution_version != policy.torch_version:
        return (
            f"torch distribution {observation.torch_distribution_version!r} does not "
            f"match policy {policy.torch_version!r}."
        )
    if observation.torch_version and observation.torch_version != policy.torch_version:
        return (
            f"torch runtime version {observation.torch_version!r} does not match "
            f"policy {policy.torch_version!r}."
        )
    if observation.torch_local_version != policy.expected_local_version:
        return (
            f"torch local-version {observation.torch_local_version!r} does not match "
            f"policy {policy.expected_local_version!r}."
        )
    expected_rocm_target = target.pytorch_rocm_target or policy.expected_local_version
    if observation.torch_rocm_target != expected_rocm_target:
        return (
            f"torch ROCm target {observation.torch_rocm_target!r} does not match "
            f"requested {expected_rocm_target!r}."
        )
    expected_hip_prefix = _rocm_major_minor(policy.expected_local_version)
    if (
        expected_hip_prefix is not None
        and observation.torch_hip_version is not None
        and not observation.torch_hip_version.startswith(expected_hip_prefix)
    ):
        return (
            f"torch HIP version {observation.torch_hip_version!r} does not match "
            f"policy {policy.expected_local_version!r}."
        )
    if observation.torchvision_distribution_version != policy.torchvision_version:
        return (
            "torchvision distribution "
            f"{observation.torchvision_distribution_version!r} does not match policy "
            f"{policy.torchvision_version!r}."
        )
    if observation.triton_rocm_status != "installed":
        return (
            f"triton-rocm status {observation.triton_rocm_status!r} does not match "
            "required installed policy."
        )
    if observation.triton_rocm_distribution_version != policy.triton_rocm_version:
        return (
            "triton-rocm distribution "
            f"{observation.triton_rocm_distribution_version!r} does not match policy "
            f"{policy.triton_rocm_version!r}."
        )
    if not _version_matches_expected(
        observation.container_rocm_user_space_version,
        policy.expected_local_version,
    ):
        return (
            "container ROCm user-space version "
            f"{observation.container_rocm_user_space_version!r} does not match policy "
            f"{policy.expected_local_version!r}."
        )
    if not _version_matches_expected(
        observation.toolchain_rocm_version,
        policy.expected_local_version,
    ):
        return (
            f"toolchain ROCm version {observation.toolchain_rocm_version!r} does not "
            f"match policy {policy.expected_local_version!r}."
        )
    return None


def _python_dependency_evidence(
    observation: PytorchDependencyObservation,
) -> MatrixPythonDependencyEvidence:
    return MatrixPythonDependencyEvidence(
        python_version=platform.python_version(),
        torch_distribution_version=observation.torch_distribution_version,
        torch_version=observation.torch_version,
        torch_local_version=observation.torch_local_version,
        torch_rocm_target=observation.torch_rocm_target,
        torch_hip_version=observation.torch_hip_version,
        torch_cuda_version=observation.torch_cuda_version,
        torch_device_available=observation.torch_device_available,
        torch_import_error=observation.torch_import_error,
        torchvision_distribution_version=observation.torchvision_distribution_version,
        triton_rocm_distribution_version=observation.triton_rocm_distribution_version,
        triton_rocm_status=observation.triton_rocm_status,
    )


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _local_version(version: str | None) -> str | None:
    if version is None or "+" not in version:
        return None
    return version.rsplit("+", maxsplit=1)[1]


def _rocm_major_minor(expected_local_version: str) -> str | None:
    if not expected_local_version.startswith("rocm"):
        return None
    parts = expected_local_version.removeprefix("rocm").split(".")
    if len(parts) < 2:
        return None
    return ".".join(parts[:2])


def _version_matches_expected(version: str | None, expected_local_version: str) -> bool:
    expected = _rocm_major_minor(expected_local_version)
    if version is None or expected is None:
        return True
    return version.startswith(expected)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser("preflight")
    preflight.add_argument(
        "--manifest", type=Path, default=DEFAULT_DOCKER_TARGET_MANIFEST
    )
    preflight.add_argument("--target")
    preflight.add_argument("--allow-mixed-version-debug", action="store_true")
    preflight.add_argument("--torch-distribution-version")
    preflight.add_argument("--torch-version")
    preflight.add_argument("--torch-local-version")
    preflight.add_argument("--torch-rocm-target")
    preflight.add_argument("--torch-hip-version")
    preflight.add_argument("--torch-cuda-version")
    preflight.add_argument("--torch-device-available", type=_parse_bool)
    preflight.add_argument("--torch-import-error")
    preflight.add_argument("--torchvision-distribution-version")
    preflight.add_argument("--triton-rocm-distribution-version")
    preflight.add_argument("--triton-rocm-status")
    preflight.add_argument("--container-rocm-user-space-version")
    preflight.add_argument("--hipcc-version")
    preflight.add_argument("--toolchain-rocm-version")
    return parser


def _observation_args_present(args: argparse.Namespace) -> bool:
    return any(
        getattr(args, name) is not None
        for name in (
            "torch_distribution_version",
            "torch_version",
            "torch_local_version",
            "torch_rocm_target",
            "torch_hip_version",
            "torch_cuda_version",
            "torch_device_available",
            "torch_import_error",
            "torchvision_distribution_version",
            "triton_rocm_distribution_version",
            "triton_rocm_status",
            "container_rocm_user_space_version",
            "hipcc_version",
            "toolchain_rocm_version",
        )
    )


def _observation_from_args(args: argparse.Namespace) -> PytorchDependencyObservation:
    return PytorchDependencyObservation(
        torch_distribution_version=_none_if_requested(args.torch_distribution_version),
        torch_version=_none_if_requested(args.torch_version),
        torch_local_version=_none_if_requested(args.torch_local_version),
        torch_rocm_target=_none_if_requested(args.torch_rocm_target),
        torch_hip_version=_none_if_requested(args.torch_hip_version),
        torch_cuda_version=_none_if_requested(args.torch_cuda_version),
        torch_device_available=args.torch_device_available,
        torch_import_error=_none_if_requested(args.torch_import_error),
        torchvision_distribution_version=_none_if_requested(
            args.torchvision_distribution_version
        ),
        triton_rocm_distribution_version=_none_if_requested(
            args.triton_rocm_distribution_version
        ),
        triton_rocm_status=_none_if_requested(args.triton_rocm_status),
        container_rocm_user_space_version=_none_if_requested(
            args.container_rocm_user_space_version
        ),
        hipcc_version=_none_if_requested(args.hipcc_version),
        toolchain_rocm_version=_none_if_requested(args.toolchain_rocm_version),
    )


def main(argv: list[str] | None = None) -> int:
    """Emit shell-consumable PyTorch ROCm dependency Matrix JSON."""

    args = _build_parser().parse_args(argv)
    if args.command == "preflight":
        selection = select_docker_target(args.target, manifest_path=args.manifest)
        policy = load_docker_target_dependency_policy(selection.target)
        observation = (
            _observation_from_args(args)
            if _observation_args_present(args)
            else collect_pytorch_dependency_observation()
        )
        payload = classify_dependency_preflight(
            target=selection.target,
            policy=policy,
            observation=observation,
            allow_mixed_version_debug=args.allow_mixed_version_debug,
        ).to_preview_payload()
        print(json.dumps(payload, sort_keys=True))
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
