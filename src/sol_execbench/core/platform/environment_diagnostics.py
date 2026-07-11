# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Environment diagnostics and lightweight smoke checks."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime

from .environment_models import (
    EnvironmentCheckResult,
    EnvironmentDiagnostics,
    EnvironmentEvidenceStatus,
    EnvironmentSnapshot,
)
from .environment_snapshot import collect_environment_snapshot


def build_environment_diagnostics(
    *,
    snapshot_collector: Callable[
        [], EnvironmentSnapshot
    ] = collect_environment_snapshot,
    smoke_checker: Callable[[], list[EnvironmentCheckResult]] | None = None,
    now: Callable[[], datetime] | None = None,
) -> EnvironmentDiagnostics:
    """Build diagnostics for ``sol-execbench environment doctor``."""

    snapshot = snapshot_collector()
    checks = tool_checks(snapshot)
    checks.extend(
        smoke_checker() if smoke_checker is not None else run_pytorch_smoke_checks()
    )
    status = aggregate_check_status(
        [snapshot.collection_status, *(check.status for check in checks)]
    )
    generated_at = (now or (lambda: datetime.now(UTC)))().isoformat()
    return EnvironmentDiagnostics(
        generated_at=generated_at,
        status=status,
        snapshot=snapshot,
        checks=checks,
    )


def run_pytorch_smoke_checks() -> list[EnvironmentCheckResult]:
    """Run lightweight PyTorch ROCm smoke checks for diagnostics only."""

    try:
        import torch
    except ImportError as exc:
        return [
            EnvironmentCheckResult(
                name="pytorch_rocm_import",
                status=EnvironmentEvidenceStatus.UNAVAILABLE,
                message=f"PyTorch unavailable: {exc}",
                remediation="Install the PyTorch ROCm wheel for this environment.",
            )
        ]

    hip_version = getattr(getattr(torch, "version", None), "hip", None)
    try:
        available = bool(torch.cuda.is_available()) and hip_version is not None
    except RuntimeError as exc:
        return [
            EnvironmentCheckResult(
                name="pytorch_rocm_runtime",
                status=EnvironmentEvidenceStatus.FAILED,
                message=f"PyTorch ROCm runtime failed: {exc}",
            )
        ]

    if not available:
        return [
            EnvironmentCheckResult(
                name="pytorch_rocm_runtime",
                status=EnvironmentEvidenceStatus.UNAVAILABLE,
                message="PyTorch ROCm device is not available",
                remediation=(
                    "Check ROCm installation, device permissions, and "
                    "HIP_VISIBLE_DEVICES."
                ),
            ),
            EnvironmentCheckResult(
                name="device_memory_copy",
                status=EnvironmentEvidenceStatus.SKIPPED,
                message="Skipped because PyTorch ROCm runtime is unavailable",
            ),
            EnvironmentCheckResult(
                name="event_timing",
                status=EnvironmentEvidenceStatus.SKIPPED,
                message="Skipped because PyTorch ROCm runtime is unavailable",
            ),
        ]

    checks = [
        EnvironmentCheckResult(
            name="pytorch_rocm_runtime",
            status=EnvironmentEvidenceStatus.AVAILABLE,
            message="PyTorch ROCm runtime is available",
        )
    ]
    try:
        tensor = torch.ones((1,), device="cuda")
        copied = tensor.to("cpu")
        if float(copied.item()) == 1.0:
            checks.append(
                EnvironmentCheckResult(
                    name="device_memory_copy",
                    status=EnvironmentEvidenceStatus.AVAILABLE,
                    message="Device memory copy succeeded",
                )
            )
        else:
            checks.append(
                EnvironmentCheckResult(
                    name="device_memory_copy",
                    status=EnvironmentEvidenceStatus.FAILED,
                    message="Device memory copy returned an unexpected value",
                )
            )
    except Exception as exc:
        checks.append(
            EnvironmentCheckResult(
                name="device_memory_copy",
                status=EnvironmentEvidenceStatus.FAILED,
                message=f"Device memory copy failed: {exc}",
            )
        )

    try:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        end.record()
        torch.cuda.synchronize()
        _ = start.elapsed_time(end)
        checks.append(
            EnvironmentCheckResult(
                name="event_timing",
                status=EnvironmentEvidenceStatus.AVAILABLE,
                message="HIP-backed PyTorch event timing succeeded",
            )
        )
    except Exception as exc:
        checks.append(
            EnvironmentCheckResult(
                name="event_timing",
                status=EnvironmentEvidenceStatus.FAILED,
                message=f"Event timing failed: {exc}",
            )
        )
    return checks


def tool_checks(snapshot: EnvironmentSnapshot) -> list[EnvironmentCheckResult]:
    checks: list[EnvironmentCheckResult] = []
    for name, result in snapshot.tools.items():
        unavailable = result.status == EnvironmentEvidenceStatus.UNAVAILABLE
        checks.append(
            EnvironmentCheckResult(
                name=f"tool:{name}",
                status=result.status,
                message=(
                    f"{name} not found on PATH"
                    if unavailable
                    else f"{name} probe {result.status.value}"
                ),
                remediation=(
                    "Install ROCm tools or ensure /opt/rocm/bin is on PATH."
                    if unavailable
                    else None
                ),
            )
        )
    return checks


def aggregate_check_status(
    statuses: list[EnvironmentEvidenceStatus],
) -> EnvironmentEvidenceStatus:
    if any(status == EnvironmentEvidenceStatus.FAILED for status in statuses):
        return EnvironmentEvidenceStatus.FAILED
    if any(status == EnvironmentEvidenceStatus.TIMEOUT for status in statuses):
        return EnvironmentEvidenceStatus.TIMEOUT
    if any(status == EnvironmentEvidenceStatus.AVAILABLE for status in statuses):
        return EnvironmentEvidenceStatus.AVAILABLE
    if any(status == EnvironmentEvidenceStatus.UNAVAILABLE for status in statuses):
        return EnvironmentEvidenceStatus.UNAVAILABLE
    return EnvironmentEvidenceStatus.SKIPPED
