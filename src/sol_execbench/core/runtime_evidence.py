"""Runtime evidence sidecars and aggregate compatibility reports."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from datetime import UTC
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import ConfigDict

from sol_execbench.core.compatibility import (
    MatrixArtifactReference,
    MatrixClaimBoundary,
    MatrixCompatibilityReasonCode,
    MatrixCompatibilityStatus,
    MatrixContainerEvidence,
    MatrixEntry,
    MatrixGpuEvidence,
    MatrixHostEvidence,
    MatrixObservedEvidence,
    MatrixToolchainEvidence,
    RocmCompatibilityMatrixReport,
    build_matrix_entry,
)
from sol_execbench.core.data.base_model import BaseModelWithDocstrings
from sol_execbench.core.dependency_matrix import (
    PytorchDependencyObservation,
    classify_dependency_preflight,
    collect_pytorch_dependency_observation,
    dependency_policy_evidence_for_target,
    load_docker_target_dependency_policy,
)
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

VISIBLE_DEVICE_ENV_VARS = (
    "HIP_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "CUDA_VISIBLE_DEVICES",
    "GPU_DEVICE_ORDINAL",
)


class RuntimeFailureEvidence(BaseModelWithDocstrings):
    """Diagnostic failure category recorded outside canonical traces."""

    model_config = _MODEL_CONFIG

    category: Literal[
        "setup_runtime",
        "dependency",
        "benchmark_correctness",
        "benchmark_performance",
    ]
    """Diagnostic evidence category."""
    status: str
    """Category-specific status value."""
    message: str | None = None
    """Optional human-readable diagnostic message."""


def collect_visible_device_environment(
    environ: dict[str, str] | None = None,
) -> dict[str, str]:
    """Collect GPU visibility environment variables when set."""

    source = os.environ if environ is None else environ
    return {name: source[name] for name in VISIBLE_DEVICE_ENV_VARS if name in source}


def collect_gpu_evidence(
    *,
    device_count: int | None = None,
    device_name: str | None = None,
    gfx_architecture: str | None = None,
    visible_device_environment: dict[str, str] | None = None,
) -> MatrixGpuEvidence:
    """Collect or build GPU evidence without requiring ROCm hardware."""

    if device_count is None or device_name is None or gfx_architecture is None:
        try:
            import torch
        except ImportError:
            pass
        else:
            try:
                if device_count is None:
                    device_count = int(torch.cuda.device_count())
                if device_name is None and device_count and device_count > 0:
                    device_name = str(torch.cuda.get_device_name(0))
                if gfx_architecture is None and device_count and device_count > 0:
                    props = torch.cuda.get_device_properties(0)
                    gfx_architecture = getattr(props, "gcnArchName", None) or getattr(
                        props, "gfx_arch", None
                    )
                    if gfx_architecture is not None:
                        gfx_architecture = str(gfx_architecture).split(":", maxsplit=1)[0]
            except (AttributeError, RuntimeError):
                pass

    return MatrixGpuEvidence(
        device_count=device_count,
        device_name=device_name,
        gfx_architecture=gfx_architecture,
        visible_device_environment=(
            collect_visible_device_environment()
            if visible_device_environment is None
            else visible_device_environment
        ),
    )


def build_host_evidence(
    *,
    rocm_version: str | None = None,
    driver_version: str | None = None,
    dev_kfd_present: bool | None = None,
    dev_kfd_accessible: bool | None = None,
    dev_dri_present: bool | None = None,
    dev_dri_accessible: bool | None = None,
    source: str = "runtime_evidence",
) -> MatrixHostEvidence:
    """Build host-scope evidence with nullable probe results."""

    device_nodes = []
    if dev_kfd_present and dev_kfd_accessible:
        device_nodes.append("/dev/kfd")
    if dev_dri_present and dev_dri_accessible:
        device_nodes.append("/dev/dri")
    return MatrixHostEvidence(
        rocm_version=rocm_version,
        driver_version=driver_version,
        device_nodes=device_nodes,
        source=source,
    )


def build_dependency_observation(
    *,
    torch_distribution_version: str | None = None,
    torch_version: str | None = None,
    torch_local_version: str | None = None,
    torch_rocm_target: str | None = None,
    torch_hip_version: str | None = None,
    torch_cuda_version: str | None = None,
    torch_device_available: bool | None = None,
    torch_import_error: str | None = None,
    torchvision_distribution_version: str | None = None,
    triton_rocm_distribution_version: str | None = None,
    triton_rocm_status: str | None = None,
    container_rocm_user_space_version: str | None = None,
    hipcc_version: str | None = None,
    toolchain_rocm_version: str | None = None,
) -> PytorchDependencyObservation:
    """Build dependency observations from injected values or local packages."""

    if not any(
        value is not None
        for value in (
            torch_distribution_version,
            torch_version,
            torch_local_version,
            torch_rocm_target,
            torch_hip_version,
            torch_cuda_version,
            torch_device_available,
            torch_import_error,
            torchvision_distribution_version,
            triton_rocm_distribution_version,
            triton_rocm_status,
        )
    ):
        observation = collect_pytorch_dependency_observation()
        updates = {
            key: value
            for key, value in {
                "container_rocm_user_space_version": container_rocm_user_space_version,
                "hipcc_version": hipcc_version,
                "toolchain_rocm_version": toolchain_rocm_version,
            }.items()
            if value is not None
        }
        return observation.model_copy(update=updates)
    return PytorchDependencyObservation(
        torch_distribution_version=torch_distribution_version,
        torch_version=torch_version,
        torch_local_version=torch_local_version,
        torch_rocm_target=torch_rocm_target,
        torch_hip_version=torch_hip_version,
        torch_cuda_version=torch_cuda_version,
        torch_device_available=torch_device_available,
        torch_import_error=torch_import_error,
        torchvision_distribution_version=torchvision_distribution_version,
        triton_rocm_distribution_version=triton_rocm_distribution_version,
        triton_rocm_status=triton_rocm_status,
        container_rocm_user_space_version=container_rocm_user_space_version,
        hipcc_version=hipcc_version,
        toolchain_rocm_version=toolchain_rocm_version,
    )


def build_runtime_matrix_entry(
    *,
    target: DockerTargetManifestEntry,
    dependency_observation: PytorchDependencyObservation,
    host: MatrixHostEvidence | None = None,
    container: MatrixContainerEvidence | None = None,
    toolchain: MatrixToolchainEvidence | None = None,
    gpu: MatrixGpuEvidence | None = None,
    runtime_unavailable_reason: str | None = None,
    failure_evidence: list[RuntimeFailureEvidence] | None = None,
    allow_mixed_version_debug: bool = False,
    container_validated: bool = False,
) -> MatrixEntry:
    """Build a diagnostic runtime Matrix Entry for one Docker Target."""

    dependency_result = classify_dependency_preflight(
        target=target,
        policy=load_docker_target_dependency_policy(target),
        observation=dependency_observation,
        allow_mixed_version_debug=allow_mixed_version_debug,
    )
    if runtime_unavailable_reason is not None:
        status = MatrixCompatibilityStatus.RUNTIME_UNAVAILABLE
        reason_code = MatrixCompatibilityReasonCode.ROCM_RUNTIME_UNAVAILABLE
        reason = runtime_unavailable_reason
    elif container_validated and (
        dependency_result.entry.status is MatrixCompatibilityStatus.NOT_TESTED
    ):
        status = MatrixCompatibilityStatus.CONTAINER_VALIDATED
        reason_code = MatrixCompatibilityReasonCode.CONTAINER_USER_SPACE_VALIDATED
        reason = (
            "Target-specific Docker wrapper benchmark completed successfully with "
            "matching container dependency and ROCm user-space evidence."
        )
    else:
        status = dependency_result.entry.status
        reason_code = dependency_result.entry.reason_code
        reason = dependency_result.entry.reason

    observed_dependency = dependency_result.entry.observed.python_dependency
    observed_policy = dependency_policy_evidence_for_target(target)
    observed_container = container or MatrixContainerEvidence(
        rocm_user_space_version=dependency_observation.container_rocm_user_space_version,
        image_repository=target.docker_image_repository,
        image_tag=target.docker_image_tag,
    )
    observed_toolchain = toolchain or MatrixToolchainEvidence(
        hipcc_version=dependency_observation.hipcc_version,
        toolchain_rocm_version=dependency_observation.toolchain_rocm_version,
    )
    artifacts = [
        MatrixArtifactReference(
            artifact_id=f"failure-{index + 1}",
            kind=f"runtime_evidence_{failure.category}",
            uri=f"diagnostic://runtime-evidence/{failure.category}/{index + 1}",
            description=failure.message or failure.status,
        )
        for index, failure in enumerate(failure_evidence or [])
    ]

    return build_matrix_entry(
        target=to_matrix_target(target),
        observed=MatrixObservedEvidence(
            host=host,
            container=observed_container,
            python_dependency=observed_dependency,
            dependency_policy=observed_policy,
            toolchain=observed_toolchain,
            gpu=gpu,
        ),
        status=status,
        reason_code=reason_code,
        reason=reason,
        claim_boundary=MatrixClaimBoundary(
            container_user_space_validated=(
                status is MatrixCompatibilityStatus.CONTAINER_VALIDATED
            ),
            native_host_validated=False,
            hardware_validated=False,
        ),
        artifacts=artifacts,
    )


def build_aggregate_report(
    entries: list[MatrixEntry],
    *,
    generated_at: str | None = None,
) -> RocmCompatibilityMatrixReport:
    """Build an aggregate compatibility matrix report from entries."""

    counts: dict[MatrixCompatibilityStatus, int] = {}
    for entry in entries:
        counts[entry.status] = counts.get(entry.status, 0) + 1
    return RocmCompatibilityMatrixReport(
        generated_at=generated_at or datetime.now(UTC).isoformat(),
        entries=entries,
        status_counts=counts,
    )


def write_json_payload(path: Path, payload: object) -> Path:
    """Write deterministic JSON with a trailing newline."""

    if hasattr(payload, "model_dump"):
        data = payload.model_dump(mode="json")
    else:
        data = payload
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, sort_keys=True) + "\n")
    return path


def write_matrix_entry(path: Path, entry: MatrixEntry) -> Path:
    """Write a per-Target Matrix Entry JSON sidecar."""

    return write_json_payload(path, entry)


def load_matrix_entry(path: Path) -> MatrixEntry:
    """Load a per-Target Matrix Entry JSON sidecar."""

    return MatrixEntry.model_validate(json.loads(path.read_text()))


def write_aggregate_report(path: Path, entries: list[MatrixEntry]) -> Path:
    """Write an aggregate compatibility matrix JSON report."""

    return write_json_payload(path, build_aggregate_report(entries))


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _none_if_requested(value: str | None) -> str | None:
    if value is None:
        return None
    if value.lower() in {"", "none", "null"}:
        return None
    return value


def _parse_bool(value: str) -> bool:
    normalized = value.lower()
    if normalized in {"1", "true", "yes"}:
        return True
    if normalized in {"0", "false", "no"}:
        return False
    raise argparse.ArgumentTypeError(f"expected boolean value, got {value!r}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect = subparsers.add_parser("collect-target")
    collect.add_argument("--manifest", type=Path, default=DEFAULT_DOCKER_TARGET_MANIFEST)
    collect.add_argument("--target")
    collect.add_argument("--output", type=Path, required=True)
    collect.add_argument("--allow-mixed-version-debug", action="store_true")
    collect.add_argument("--host-rocm-version")
    collect.add_argument("--host-driver-version")
    collect.add_argument("--dev-kfd-present", type=_parse_bool)
    collect.add_argument("--dev-kfd-accessible", type=_parse_bool)
    collect.add_argument("--dev-dri-present", type=_parse_bool)
    collect.add_argument("--dev-dri-accessible", type=_parse_bool)
    collect.add_argument("--image-digest")
    collect.add_argument("--torch-distribution-version")
    collect.add_argument("--torch-version")
    collect.add_argument("--torch-local-version")
    collect.add_argument("--torch-rocm-target")
    collect.add_argument("--torch-hip-version")
    collect.add_argument("--torch-cuda-version")
    collect.add_argument("--torch-device-available", type=_parse_bool)
    collect.add_argument("--torch-import-error")
    collect.add_argument("--torchvision-distribution-version")
    collect.add_argument("--triton-rocm-distribution-version")
    collect.add_argument("--triton-rocm-status")
    collect.add_argument("--container-rocm-user-space-version")
    collect.add_argument("--hipcc-version")
    collect.add_argument("--toolchain-rocm-version")
    collect.add_argument("--device-count", type=int)
    collect.add_argument("--device-name")
    collect.add_argument("--gfx-architecture")
    collect.add_argument("--visible-device-env", action="append", default=[])
    collect.add_argument("--runtime-unavailable-reason")
    collect.add_argument("--container-validated", action="store_true")
    collect.add_argument(
        "--failure-category",
        action="append",
        choices=[
            "setup_runtime",
            "dependency",
            "benchmark_correctness",
            "benchmark_performance",
        ],
        default=[],
    )

    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--output", type=Path, required=True)
    aggregate.add_argument("entries", type=Path, nargs="+")
    return parser


def _visible_env_from_args(values: list[str]) -> dict[str, str]:
    result = collect_visible_device_environment()
    for value in values:
        if "=" not in value:
            raise argparse.ArgumentTypeError(
                f"expected NAME=VALUE visible device env, got {value!r}"
            )
        key, env_value = value.split("=", maxsplit=1)
        result[key] = env_value
    return result


def _failure_evidence_from_args(categories: list[str]) -> list[RuntimeFailureEvidence]:
    return [
        RuntimeFailureEvidence(category=category, status="recorded")
        for category in categories
    ]


def _collect_target(args: argparse.Namespace) -> MatrixEntry:
    selection = select_docker_target(args.target, manifest_path=args.manifest)
    dependency_observation = build_dependency_observation(
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
    return build_runtime_matrix_entry(
        target=selection.target,
        dependency_observation=dependency_observation,
        host=build_host_evidence(
            rocm_version=_none_if_requested(args.host_rocm_version),
            driver_version=_none_if_requested(args.host_driver_version),
            dev_kfd_present=args.dev_kfd_present,
            dev_kfd_accessible=args.dev_kfd_accessible,
            dev_dri_present=args.dev_dri_present,
            dev_dri_accessible=args.dev_dri_accessible,
        ),
        container=MatrixContainerEvidence(
            rocm_user_space_version=dependency_observation.container_rocm_user_space_version,
            image_repository=selection.target.docker_image_repository,
            image_tag=selection.target.docker_image_tag,
            image_digest=_none_if_requested(args.image_digest),
        ),
        gpu=collect_gpu_evidence(
            device_count=args.device_count,
            device_name=_none_if_requested(args.device_name),
            gfx_architecture=_none_if_requested(args.gfx_architecture),
            visible_device_environment=_visible_env_from_args(args.visible_device_env),
        ),
        runtime_unavailable_reason=_none_if_requested(args.runtime_unavailable_reason),
        failure_evidence=_failure_evidence_from_args(args.failure_category),
        allow_mixed_version_debug=args.allow_mixed_version_debug,
        container_validated=args.container_validated,
    )


def main(argv: list[str] | None = None) -> int:
    """Emit runtime evidence sidecars and aggregate reports."""

    args = _build_parser().parse_args(argv)
    if args.command == "collect-target":
        entry = _collect_target(args)
        write_matrix_entry(args.output, entry)
        print(json.dumps(entry.model_dump(mode="json"), sort_keys=True))
        return 0
    if args.command == "aggregate":
        report = build_aggregate_report([load_matrix_entry(path) for path in args.entries])
        write_json_payload(args.output, report)
        print(json.dumps(report.model_dump(mode="json"), sort_keys=True))
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
