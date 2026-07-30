"""Runtime evidence collection helpers."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Literal

from sol_execbench.core.evidence.runtime_evidence.models import (
    RuntimeGPUTelemetry,
)
from sol_execbench.core.platform.amd_smi import (
    parse_gpu_identity,
    parse_gpu_metrics,
    parse_performance_levels,
    parse_processes,
)
from sol_execbench.core.platform.compatibility import (
    MatrixGPUEvidence,
    MatrixHostEvidence,
)
from sol_execbench.core.platform.dependency_matrix import (
    PytorchDependencyObservation,
    collect_pytorch_dependency_observation,
)
from sol_execbench.core.platform.runtime import resolve_rocm_tool
from sol_execbench.core.process.subprocesses import run_in_process_group_bounded

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
    return {
        name: source[name] for name in VISIBLE_DEVICE_ENV_VARS if name in source
    }


def collect_gpu_evidence(
    *,
    device_count: int | None = None,
    device_name: str | None = None,
    gfx_architecture: str | None = None,
    visible_device_environment: dict[str, str] | None = None,
) -> MatrixGPUEvidence:
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
                if (
                    gfx_architecture is None
                    and device_count
                    and device_count > 0
                ):
                    props = torch.cuda.get_device_properties(0)
                    gfx_architecture = getattr(
                        props,
                        "gcnArchName",
                        None,
                    ) or getattr(
                        props,
                        "gfx_arch",
                        None,
                    )
                    if gfx_architecture is not None:
                        gfx_architecture = str(gfx_architecture).split(
                            ":",
                            maxsplit=1,
                        )[0]
            except (AttributeError, RuntimeError):
                pass

    return MatrixGPUEvidence(
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
    collect_observation: Callable[[], PytorchDependencyObservation]
    | None = None,
) -> PytorchDependencyObservation:
    """Build dependency observations from injected values or local packages."""
    overrides = overrides or PytorchDependencyObservation()
    if not any(
        getattr(overrides, field) is not None for field in _LOCAL_RUNTIME_FIELDS
    ):
        collector = (
            collect_observation or collect_pytorch_dependency_observation
        )
        observation = collector()
        updates = {
            field: value
            for field in _TOOLCHAIN_OVERRIDE_FIELDS
            if (value := getattr(overrides, field)) is not None
        }
        return observation.model_copy(update=updates)
    return overrides


def collect_runtime_gpu_telemetry(
    *,
    phase: Literal["pre", "post"],
    device_index: int = 0,
) -> RuntimeGPUTelemetry:
    """Collect a bounded pre/post AMD SMI snapshot for replay admission."""
    amd_smi = resolve_rocm_tool("amd-smi")
    if amd_smi is None:
        return RuntimeGPUTelemetry(phase=phase)
    identity_raw = _amd_smi_json(amd_smi, "list")
    metric_raw = _amd_smi_json(amd_smi, "metric")
    process_raw = _amd_smi_json(amd_smi, "process")
    if identity_raw is None or metric_raw is None:
        return RuntimeGPUTelemetry(phase=phase)
    identity = parse_gpu_identity(identity_raw, device_index)
    metrics = parse_gpu_metrics(metric_raw, device_index)
    levels = parse_performance_levels(metric_raw)
    processes = parse_processes(process_raw) if process_raw is not None else []
    foreign = sum(process["pid"] != os.getpid() for process in processes)
    return RuntimeGPUTelemetry(
        phase=phase,
        gpu_id=identity.uuid,
        gpu_bdf=identity.bdf,
        performance_level=(
            levels[device_index] if device_index < len(levels) else None
        ),
        sclk_mhz=metrics.sclk_mhz,
        mclk_mhz=metrics.mclk_mhz,
        temperature_c=metrics.temperature_c,
        power_profile=metrics.power_profile,
        power_cap_w=metrics.power_cap_w,
        power_draw_w=metrics.power_draw_w,
        foreign_process_count=foreign,
    )


def _amd_smi_json(executable: Path, command: str) -> str | None:
    completed = run_in_process_group_bounded(
        [str(executable), command, "--json"],
        timeout=10.0,
    )
    return completed.stdout if completed.returncode == 0 else None
