# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Environment snapshot collection and summarization."""

from __future__ import annotations

import os
import shutil
from collections.abc import Callable
from datetime import UTC, datetime

from .arch_capabilities import (
    ArchCapabilityBudgetStatus,
    derive_arch_capability_budget,
)
from .environment_models import (
    DEFAULT_PROBE_TIMEOUT_SECONDS,
    EnvironmentCapabilityBudget,
    EnvironmentEvidenceStatus,
    EnvironmentSnapshot,
    GpuEnvironmentSummary,
    ProbeRunner,
    PytorchRocmSummary,
    RocmEnvironmentSummary,
    ToolProbeResult,
    Which,
)
from .environment_probes import collect_pytorch_rocm_summary, probe_tool, run_probe


def collect_environment_snapshot(
    *,
    runner: ProbeRunner | None = None,
    which: Which = shutil.which,
    timeout_seconds: float = DEFAULT_PROBE_TIMEOUT_SECONDS,
    collect_pytorch: bool = True,
    now: Callable[[], datetime] | None = None,
) -> EnvironmentSnapshot:
    """Collect optional ROCm environment evidence."""

    effective_runner = runner or run_probe
    generated_at = (now or (lambda: datetime.now(UTC)))().isoformat()
    tools = {
        "amd-smi": probe_tool(
            "amd-smi",
            ["amd-smi", "static", "-a"],
            runner=effective_runner,
            which=which,
            timeout_seconds=timeout_seconds,
        ),
        "rocminfo": probe_tool(
            "rocminfo",
            ["rocminfo"],
            runner=effective_runner,
            which=which,
            timeout_seconds=timeout_seconds,
        ),
        "rocm_agent_enumerator": probe_tool(
            "rocm_agent_enumerator",
            ["rocm_agent_enumerator"],
            runner=effective_runner,
            which=which,
            timeout_seconds=timeout_seconds,
        ),
    }
    pytorch = collect_pytorch_rocm_summary() if collect_pytorch else None
    visible_devices = visible_device_environment()
    gpus = summarize_gpus(tools, pytorch)
    capability_budgets = derive_capability_budgets(gpus)
    warnings = snapshot_warnings(tools, pytorch)
    return EnvironmentSnapshot(
        generated_at=generated_at,
        collection_status=aggregate_status(tools, pytorch),
        tools=tools,
        gpus=gpus,
        capability_budgets=capability_budgets,
        rocm=RocmEnvironmentSummary(
            hip_visible_devices=visible_devices.get("HIP_VISIBLE_DEVICES"),
            rocr_visible_devices=visible_devices.get("ROCR_VISIBLE_DEVICES"),
            hsa_override_gfx_version=visible_devices.get("HSA_OVERRIDE_GFX_VERSION"),
        ),
        pytorch=pytorch,
        visible_devices=visible_devices,
        warnings=warnings,
    )


def aggregate_status(
    tools: dict[str, ToolProbeResult],
    pytorch: PytorchRocmSummary | None,
) -> EnvironmentEvidenceStatus:
    statuses = {tool.status for tool in tools.values()}
    if pytorch and pytorch.available:
        return EnvironmentEvidenceStatus.AVAILABLE
    if EnvironmentEvidenceStatus.AVAILABLE in statuses:
        return EnvironmentEvidenceStatus.AVAILABLE
    if EnvironmentEvidenceStatus.FAILED in statuses:
        return EnvironmentEvidenceStatus.FAILED
    if EnvironmentEvidenceStatus.TIMEOUT in statuses:
        return EnvironmentEvidenceStatus.TIMEOUT
    return EnvironmentEvidenceStatus.UNAVAILABLE


def summarize_gpus(
    tools: dict[str, ToolProbeResult],
    pytorch: PytorchRocmSummary | None,
) -> list[GpuEnvironmentSummary]:
    gpus: list[GpuEnvironmentSummary] = []
    # Probe tools expose ISA targets rather than device IDs.  If PyTorch sees a
    # single device, a matching target from a tool is evidence about that device,
    # not an additional GPU.
    pytorch_targets = (
        {pytorch.gfx_target.lower()}
        if pytorch is not None and pytorch.device_count == 1 and pytorch.gfx_target
        else set()
    )
    if pytorch and (pytorch.device_name or pytorch.gfx_target):
        gpus.append(
            GpuEnvironmentSummary(
                source="pytorch",
                index=0,
                name=pytorch.device_name,
                gfx_target=pytorch.gfx_target,
            )
        )
    for tool_name, result in tools.items():
        gfx_targets = result.parsed.get("gfx_targets")
        if isinstance(gfx_targets, list):
            for index, gfx_target in enumerate(gfx_targets):
                if str(gfx_target).lower() in pytorch_targets:
                    continue
                gpus.append(
                    GpuEnvironmentSummary(
                        source=tool_name,
                        index=index,
                        gfx_target=str(gfx_target),
                    )
                )
    return gpus


def derive_capability_budgets(
    gpus: list[GpuEnvironmentSummary],
) -> list[EnvironmentCapabilityBudget]:
    """Derive arch capability budgets for detected GPUs.

    Each distinct gfx architecture yields one budget entry. Uncovered
    architectures downgrade to ``unsupported`` rather than promoting unknown
    budget values.
    """

    budgets: list[EnvironmentCapabilityBudget] = []
    seen: set[str] = set()
    for gpu in gpus:
        gfx_target = gpu.gfx_target
        if gfx_target is None:
            continue
        token = gfx_target.split(":", maxsplit=1)[0].strip().lower()
        if token in seen:
            continue
        seen.add(token)
        budget = derive_arch_capability_budget(gfx_target)
        if budget is not None:
            budgets.append(
                EnvironmentCapabilityBudget(
                    status=ArchCapabilityBudgetStatus.AVAILABLE,
                    architecture=budget.architecture,
                    budget=budget,
                    source="packaged:arch_capability_budgets",
                )
            )
        else:
            budgets.append(
                EnvironmentCapabilityBudget(
                    status=ArchCapabilityBudgetStatus.UNSUPPORTED,
                    architecture=gfx_target,
                    reason_code="unsupported_architecture",
                    source="packaged:arch_capability_budgets",
                )
            )
    return budgets


def snapshot_warnings(
    tools: dict[str, ToolProbeResult],
    pytorch: PytorchRocmSummary | None,
) -> list[str]:
    warnings: list[str] = []
    for name, result in tools.items():
        if result.status != EnvironmentEvidenceStatus.AVAILABLE:
            warnings.append(f"{name}:{result.status.value}")
    if pytorch and not pytorch.available:
        warnings.append("pytorch_rocm:unavailable")
    return warnings


def visible_device_environment() -> dict[str, str]:
    keys = ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "HSA_OVERRIDE_GFX_VERSION")
    return {key: value for key in keys if (value := os.environ.get(key)) is not None}
