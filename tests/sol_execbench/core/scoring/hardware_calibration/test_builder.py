# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

import pytest

from sol_execbench.core.scoring.hardware_calibration.builder import (
    CalibrationRequest,
    ClockLockRequiredError,
    run_calibration,
)
from sol_execbench.core.scoring.hardware_calibration.environment import GpuEnvironment
from sol_execbench.core.scoring.hardware_calibration.hip_probe import (
    HipProbe,
    ProbeExecution,
)


def _probe() -> HipProbe:
    return HipProbe(
        compile_candidate=lambda _: "passed",
        execute_candidate=lambda key: ProbeExecution(
            (10.0,), "TFLOP/s" if key.kind == "compute" else "GB/s"
        ),
        check_correctness=lambda _, __: True,
        check_stability=lambda _, __: True,
    )


def test_missing_clock_adapter_is_collected_but_provisional() -> None:
    artifact = run_calibration(
        CalibrationRequest(
            environment=GpuEnvironment(device=0, architecture="gfx942"),
            hip_probe=_probe(),
        )
    )

    assert artifact.collection_status == "collected"
    assert artifact.validation_status == "provisional"
    assert artifact.metadata["clock_lock_reason"] == "clock_lock_adapter_unavailable"


def test_require_clock_lock_rejects_missing_adapter() -> None:
    with pytest.raises(ClockLockRequiredError):
        run_calibration(
            CalibrationRequest(
                environment=GpuEnvironment(device=0, architecture="gfx950"),
                hip_probe=_probe(),
                require_clock_lock=True,
            )
        )
