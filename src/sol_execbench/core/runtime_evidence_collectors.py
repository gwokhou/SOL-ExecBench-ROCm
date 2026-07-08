"""Runtime evidence collection helpers."""

from __future__ import annotations

import os
from collections.abc import Callable

from sol_execbench.core.compatibility import MatrixGpuEvidence, MatrixHostEvidence
from sol_execbench.core.dependency_matrix import (
    PytorchDependencyObservation,
    collect_pytorch_dependency_observation,
)


VISIBLE_DEVICE_ENV_VARS = (
    "HIP_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "CUDA_VISIBLE_DEVICES",
    "GPU_DEVICE_ORDINAL",
)


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
                        gfx_architecture = str(gfx_architecture).split(":", maxsplit=1)[
                            0
                        ]
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
    collect_observation: Callable[[], PytorchDependencyObservation] | None = None,
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
        collector = collect_observation or collect_pytorch_dependency_observation
        observation = collector()
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
