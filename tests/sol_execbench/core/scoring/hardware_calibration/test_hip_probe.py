# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

from sol_execbench.core.scoring.hardware_calibration.hip_probe import (
    CalibrationProfileKey,
    HipCommandBackend,
    HipProbe,
    ProbeExecution,
    _hip_source,
    default_hip_probe,
)


def test_default_probe_passes_architecture_to_command_backend(monkeypatch) -> None:
    backend = HipCommandBackend(hipcc=None, architecture="gfx1200")

    monkeypatch.setattr(
        "sol_execbench.core.scoring.hardware_calibration.hip_probe.HipCommandBackend",
        lambda *, architecture: backend,
    )

    default_hip_probe(architecture="gfx1200")

    assert backend.architecture == "gfx1200"


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


def test_malformed_execution_output_is_unknown() -> None:
    key = CalibrationProfileKey("compute", "vector", "fp32", "fp32", "portable")
    probe = HipProbe(
        compile_candidate=lambda _: "passed",
        execute_candidate=lambda _: object(),  # type: ignore[arg-type]
        check_correctness=lambda _, __: True,
        check_stability=lambda _, __: True,
    )

    candidate = probe.measure(key)

    assert candidate.state == "unknown"
    assert candidate.reason_code == "hip_probe_execution_malformed"


def test_default_backend_hipcc_failure_is_unknown(tmp_path) -> None:
    probe = default_hip_probe(
        HipCommandBackend(workspace=tmp_path, hipcc="/missing/hipcc")
    )
    key = CalibrationProfileKey("compute", "vector", "fp32", "fp32", "portable")

    candidate = probe.measure(key)

    assert candidate.state == "unknown"
    assert candidate.reason_code == "hip_probe_compile_failed"


def test_matrix_path_mismatch_is_explicitly_unavailable(tmp_path) -> None:
    backend = HipCommandBackend(
        workspace=tmp_path, hipcc="hipcc", architecture="gfx942"
    )
    key = CalibrationProfileKey("compute", "matrix", "bf16", "bf16", "wmma")

    assert backend.compile(key) == "unsupported"


def test_missing_compiler_is_unknown_not_unavailable(tmp_path) -> None:
    probe = default_hip_probe(
        HipCommandBackend(workspace=tmp_path, hipcc=None, architecture="gfx942")
    )
    key = CalibrationProfileKey("compute", "matrix", "bf16", "bf16", "mfma")

    candidate = probe.measure(key)

    assert candidate.state == "unknown"
    assert candidate.reason_code == "hip_probe_compile_missing"


def test_wmma_source_uses_bf16_wmma_intrinsic() -> None:
    source = _hip_source(
        CalibrationProfileKey("compute", "matrix", "bf16", "bf16", "wmma")
    )

    assert "__hip_bfloat16" in source
    assert "wmma" in source.lower()
    assert "RESULT" in source


def test_mfma_source_uses_bf16_mfma_intrinsic() -> None:
    source = _hip_source(
        CalibrationProfileKey("compute", "matrix", "bf16", "bf16", "mfma")
    )

    assert "__hip_bfloat16" in source
    assert "mfma" in source.lower()
    assert "RESULT" in source


def test_matrix_source_validates_before_timed_samples_and_result_output() -> None:
    source = _hip_source(
        CalibrationProfileKey("compute", "matrix", "bf16", "bf16", "wmma")
    )

    validation_launch = source.index(
        "hipLaunchKernelGGL(matrix_probe_kernel, dim3(1), dim3(lane_count), 0, 0, device_output);"
    )
    validation_sync = source.index("hipDeviceSynchronize()", validation_launch)
    validation_copy = source.index("hipMemcpy(output.data(), device_output", validation_sync)
    validation_check = source.index("std::fabs(value - cpu_reference)", validation_copy)
    validation_failure = source.index("return 4", validation_check)
    timed_loop = source.index("for (int sample = 0; sample < 7; ++sample)")
    result_output = source.index('std::printf("RESULT')

    assert (
        validation_launch
        < validation_sync
        < validation_copy
        < validation_check
        < validation_failure
        < timed_loop
    )
    assert validation_failure < result_output


def test_matrix_compile_targets_selected_architecture(tmp_path) -> None:
    commands: list[list[str]] = []
    backend = HipCommandBackend(
        workspace=tmp_path,
        hipcc="hipcc",
        architecture="gfx1200",
        run=lambda command, **_: commands.append(command)
        or type("Result", (), {"returncode": 0})(),
    )

    assert (
        backend.compile(
            CalibrationProfileKey("compute", "matrix", "bf16", "bf16", "wmma")
        )
        == "passed"
    )
    assert "--offload-arch=gfx1200" in commands[0]
