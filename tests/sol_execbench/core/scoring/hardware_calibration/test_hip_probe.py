# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

from sol_execbench.core.scoring.hardware_calibration.hip_probe import (
    CalibrationProfileKey,
    HipProbe,
    ProbeExecution,
)


def test_explicitly_unsupported_probe_is_unavailable() -> None:
    key = CalibrationProfileKey("compute", "vector", "fp32", "fp32", "portable")
    probe = HipProbe(
        compile_candidate=lambda _: "unsupported",
        execute_candidate=lambda _: ProbeExecution((1.0,), "TFLOP/s"),
        check_correctness=lambda _, __: True,
        check_stability=lambda _, __: True,
    )

    candidate = probe.measure(key)

    assert candidate.state == "unavailable"
    assert candidate.reason_code == "hip_probe_explicitly_unsupported"


def test_failed_probe_is_unknown() -> None:
    key = CalibrationProfileKey("memory", "stream_copy", "fp32", "fp32", "portable")
    probe = HipProbe(
        compile_candidate=lambda _: "passed",
        execute_candidate=lambda _: ProbeExecution((1.0,), "GB/s"),
        check_correctness=lambda _, __: False,
        check_stability=lambda _, __: True,
    )

    candidate = probe.measure(key)

    assert candidate.state == "unknown"
    assert candidate.reason_code == "hip_probe_correctness_failed"
