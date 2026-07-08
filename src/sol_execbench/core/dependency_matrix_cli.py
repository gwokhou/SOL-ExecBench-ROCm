"""PyTorch ROCm dependency policy helpers for declared Docker Targets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sol_execbench.core.utils import (
    none_if_requested as _none_if_requested,
    parse_bool as _parse_bool,
)
from sol_execbench.core.dependency_matrix_classification import classify_dependency_preflight
from sol_execbench.core.dependency_matrix_collectors import collect_pytorch_dependency_observation
from sol_execbench.core.dependency_matrix_models import PytorchDependencyObservation
from sol_execbench.core.dependency_matrix_policy import load_docker_target_dependency_policy
from sol_execbench.core.docker_matrix import (
    DEFAULT_DOCKER_TARGET_MANIFEST,
    select_docker_target,
)


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
