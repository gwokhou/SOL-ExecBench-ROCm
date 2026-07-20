"""Runtime evidence collection helpers."""

from __future__ import annotations

import os
from collections.abc import Callable

from sol_execbench.core.platform.compatibility import (
    MatrixGpuEvidence,
    MatrixHostEvidence,
)
from sol_execbench.core.platform.dependency_matrix import (
    PytorchDependencyObservation,
    collect_pytorch_dependency_observation,
)


VISIBLE_DEVICE_ENV_VARS = (
    "HIP_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "CUDA_VISIBLE_DEVICES",
    "GPU_DEVICE_ORDINAL",
)
_LOCAL_RUNTIME_FIELDS = (
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
)
_TOOLCHAIN_OVERRIDE_FIELDS = (
    "container_rocm_user_space_version",
    "hipcc_version",
    "toolchain_rocm_version",
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
    overrides: PytorchDependencyObservation | None = None,
    *,
    collect_observation: Callable[[], PytorchDependencyObservation] | None = None,
) -> PytorchDependencyObservation:
    """Build dependency observations from injected values or local packages."""
    overrides = overrides or PytorchDependencyObservation()
    if not any(
        getattr(overrides, field) is not None for field in _LOCAL_RUNTIME_FIELDS
    ):
        collector = collect_observation or collect_pytorch_dependency_observation
        observation = collector()
        updates = {
            field: value
            for field in _TOOLCHAIN_OVERRIDE_FIELDS
            if (value := getattr(overrides, field)) is not None
        }
        return observation.model_copy(update=updates)
    return overrides
