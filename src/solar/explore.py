# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Public fixed-instance exploration over complete, pinned SOLAR analyses."""

from solar.analysis.exploration_contracts import InverseRequest, InverseResult
from solar.analysis.inverse import solve_constraints
from solar.analysis.scan_contracts import ScanResult
from solar.pipeline.exploration import model_from_projection, read_projection
from solar.scan import ScanExecutor
from solar.scan_contracts import ScanRequest


def query_resources(request: InverseRequest) -> InverseResult:
    """Return necessary resource constraints for the caller's fixed-capacity model."""
    result = read_projection(request.source)
    return solve_constraints(
        model_from_projection(result, request.view, request.computation),
        target_seconds=request.target_seconds,
        scale_resources=request.scale_resources,
    )


def scan_profiles(
    request: ScanRequest, *, execution_identity: dict[str, str] | None = None
) -> ScanResult:
    """Run one serial CPU analysis attempt; all profile points are hypothetical.

    The in-process API requires the Linux main thread and no active real-time
    signal timer. The CLI supplies process isolation and its own runtime identity.
    """
    return ScanExecutor(
        request, execution_identity or {"runtime": "in_process"}
    ).run()
