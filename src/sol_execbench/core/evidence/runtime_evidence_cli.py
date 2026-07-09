"""Command-line interface for runtime evidence sidecars."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

from sol_execbench.core.platform.compatibility import (
    MatrixContainerEvidence,
    MatrixEntry,
)
from sol_execbench.core.platform.dependency_matrix import PytorchDependencyObservation
from sol_execbench.core.platform.docker_matrix import (
    DEFAULT_DOCKER_TARGET_MANIFEST,
    select_docker_target,
)
from sol_execbench.core.evidence.runtime_evidence_builders import (
    build_aggregate_report,
    build_runtime_matrix_entry,
)
from sol_execbench.core.evidence.runtime_evidence_collectors import (
    build_host_evidence,
    collect_gpu_evidence,
    collect_visible_device_environment,
)
from sol_execbench.core.evidence.runtime_evidence_io import (
    load_matrix_entry,
    write_json_payload,
    write_matrix_entry,
)
from sol_execbench.core.evidence.runtime_evidence_models import (
    RuntimeFailureCategory,
    RuntimeFailureEvidence,
)
from sol_execbench.core.utils import (
    none_if_requested as _none_if_requested,
    parse_bool as _parse_bool,
)


DependencyObservationBuilder = Callable[..., PytorchDependencyObservation]


def build_parser() -> argparse.ArgumentParser:
    """Build the runtime evidence CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect = subparsers.add_parser("collect-target")
    collect.add_argument(
        "--manifest", type=Path, default=DEFAULT_DOCKER_TARGET_MANIFEST
    )
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


def visible_env_from_args(values: list[str]) -> dict[str, str]:
    """Merge current visible-device env with explicit NAME=VALUE CLI values."""
    result = collect_visible_device_environment()
    for value in values:
        if "=" not in value:
            raise argparse.ArgumentTypeError(
                f"expected NAME=VALUE visible device env, got {value!r}"
            )
        key, env_value = value.split("=", maxsplit=1)
        result[key] = env_value
    return result


def failure_evidence_from_args(
    categories: list[RuntimeFailureCategory],
) -> list[RuntimeFailureEvidence]:
    """Build recorded failure evidence from CLI category values."""
    return [
        RuntimeFailureEvidence(category=category, status="recorded")
        for category in categories
    ]


def collect_target(
    args: argparse.Namespace,
    *,
    build_dependency_observation: DependencyObservationBuilder,
) -> MatrixEntry:
    """Collect one target entry from parsed CLI arguments."""
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
            visible_device_environment=visible_env_from_args(args.visible_device_env),
        ),
        runtime_unavailable_reason=_none_if_requested(args.runtime_unavailable_reason),
        failure_evidence=failure_evidence_from_args(args.failure_category),
        allow_mixed_version_debug=args.allow_mixed_version_debug,
        container_validated=args.container_validated,
    )


def main(
    argv: list[str] | None = None,
    *,
    build_dependency_observation: DependencyObservationBuilder,
) -> int:
    """Emit runtime evidence sidecars and aggregate reports."""
    args = build_parser().parse_args(argv)
    if args.command == "collect-target":
        entry = collect_target(
            args, build_dependency_observation=build_dependency_observation
        )
        write_matrix_entry(args.output, entry)
        print(json.dumps(entry.model_dump(mode="json"), sort_keys=True))
        return 0
    if args.command == "aggregate":
        report = build_aggregate_report(
            [load_matrix_entry(path) for path in args.entries]
        )
        write_json_payload(args.output, report)
        print(json.dumps(report.model_dump(mode="json"), sort_keys=True))
        return 0
    raise AssertionError(f"unhandled command: {args.command}")
