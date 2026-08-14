# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Exact extraction, conversion, and replay audit for corpus workloads."""

from __future__ import annotations

from pathlib import Path

from sol_execbench.core.solar_bridge.formal_device import (
    FORMAL_ARCHITECTURE,
    require_formal_device,
)
from sol_execbench.core.solar_bridge.models import SolarStageAuditOutcome
from sol_execbench.core.solar_bridge.workload_context import (
    load_solar_workload_context,
    solar_conversion_request,
)
from solar.ir.contracts import DEFAULT_IR_PATH, IRPath, normalize_ir_path


def audit_workload_stages(
    *,
    problem_dir: str | Path,
    workload_uuid: str,
    output_dir: str | Path,
    device: str,
    ir_path: IRPath | str = DEFAULT_IR_PATH,
) -> SolarStageAuditOutcome:
    """Run the exact extraction/conversion/replay gate for one workload."""
    from solar.api import ConversionReadinessRequest, audit_conversion

    require_formal_device(device)
    selected_path = normalize_ir_path(ir_path)
    context = load_solar_workload_context(problem_dir, workload_uuid, device)
    result = audit_conversion(
        ConversionReadinessRequest(
            conversion=solar_conversion_request(
                context,
                device,
                selected_path,
            ),
            architecture=FORMAL_ARCHITECTURE,
            output_dir=Path(output_dir),
        ),
    )
    return SolarStageAuditOutcome.from_dict(result.to_dict())


__all__ = ["audit_workload_stages"]
