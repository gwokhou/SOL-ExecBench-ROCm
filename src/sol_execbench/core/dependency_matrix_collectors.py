"""PyTorch ROCm dependency policy helpers for declared Docker Targets."""

from __future__ import annotations

import importlib.metadata
import subprocess
from pathlib import Path

from sol_execbench.core.dependency_matrix_models import PytorchDependencyObservation


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


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _local_version(version: str | None) -> str | None:
    if version is None or "+" not in version:
        return None
    return version.rsplit("+", maxsplit=1)[1]
