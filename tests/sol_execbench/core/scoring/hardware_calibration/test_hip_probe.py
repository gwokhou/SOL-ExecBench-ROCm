# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

import hashlib
from collections.abc import Callable
from importlib import resources
from pathlib import Path
from typing import cast

import pytest

import sol_execbench.core.scoring.hardware_calibration.hip_probe as hip_probe_module
from sol_execbench.core.scoring.hardware_calibration.hip_probe import (
    CalibrationProfileKey,
    HipCommandBackend,
    HipProbe,
    ProbeExecution,
    _hip_source,
    default_hip_probe,
)


_RESOURCE_CASES = (
    (
        "vector_fp32_fp32.hip",
        CalibrationProfileKey("compute", "vector", "fp32", "fp32", "portable"),
    ),
    (
        "stream_copy_fp32_fp32.hip",
        CalibrationProfileKey("memory", "stream_copy", "fp32", "fp32", "portable"),
    ),
    (
        "vector_fp16_fp16.hip",
        CalibrationProfileKey("compute", "vector", "fp16", "fp16", "gfx12"),
    ),
    (
        "stream_copy_fp16_fp16.hip",
        CalibrationProfileKey("memory", "stream_copy", "fp16", "fp16", "gfx12"),
    ),
    (
        "vector_bf16_bf16.hip",
        CalibrationProfileKey("compute", "vector", "bf16", "bf16", "gfx12"),
    ),
    (
        "stream_copy_bf16_bf16.hip",
        CalibrationProfileKey("memory", "stream_copy", "bf16", "bf16", "gfx12"),
    ),
    (
        "stream_copy_bf16_fp32.hip",
        CalibrationProfileKey("memory", "stream_copy", "bf16", "fp32", "gfx12"),
    ),
    (
        "stream_copy_fp32_bf16.hip",
        CalibrationProfileKey("memory", "stream_copy", "fp32", "bf16", "gfx12"),
    ),
    (
        "reduction_fp32_fp32.hip",
        CalibrationProfileKey("compute", "reduction", "fp32", "fp32", "gfx12"),
    ),
    (
        "transcendental_fp32_fp32.hip",
        CalibrationProfileKey("compute", "transcendental", "fp32", "fp32", "gfx12"),
    ),
    (
        "matrix_bf16_bf16_wmma.hip",
        CalibrationProfileKey("compute", "matrix", "bf16", "bf16", "wmma"),
    ),
    (
        "matrix_fp16_fp16_wmma.hip",
        CalibrationProfileKey("compute", "matrix", "fp16", "fp16", "wmma"),
    ),
    (
        "matrix_bf16_bf16_mfma.hip",
        CalibrationProfileKey("compute", "matrix", "bf16", "bf16", "mfma"),
    ),
)


def test_fp32_matrix_profile_reuses_the_gfx12_fma_probe() -> None:
    key = CalibrationProfileKey("compute", "matrix", "fp32", "fp32", "gfx12")

    assert _hip_source(key) == _hip_source(
        CalibrationProfileKey("compute", "vector", "fp32", "fp32", "portable")
    )
    assert "fmaf(value" in _hip_source(key)


@pytest.mark.parametrize(("filename", "key"), _RESOURCE_CASES)
def test_every_probe_route_loads_its_package_resource(
    filename: str, key: CalibrationProfileKey
) -> None:
    resource = resources.files(
        "sol_execbench.data.hardware_calibration_probes"
    ).joinpath(filename)

    assert resource.is_file()
    assert _hip_source(key) == resource.read_text(encoding="utf-8")
    suffix = f"_{key.path}" if key.operation == "matrix" else ""
    assert filename == (
        f"{key.operation}_{key.input_dtype}_{key.output_dtype}{suffix}.hip"
    )


def test_probe_resources_are_complete_and_utf8() -> None:
    root = resources.files("sol_execbench.data.hardware_calibration_probes")
    expected = {filename for filename, _ in _RESOURCE_CASES}
    packaged = {item.name for item in root.iterdir() if item.name.endswith(".hip")}

    assert packaged == expected
    assert all(
        root.joinpath(filename).read_text(encoding="utf-8") for filename in expected
    )


@pytest.mark.parametrize(
    ("filename", "expected_sha256"),
    (
        (
            "vector_fp32_fp32.hip",
            "ba990fd967177b0d3d32640a3f8779ec4c695dc7e34645477f02b200aa63e2f3",
        ),
        (
            "stream_copy_bf16_fp32.hip",
            "042bfe2390fb532e7a8cc5ab32e0fbebe2944c6bdadfe5c580659f187a829966",
        ),
        (
            "matrix_bf16_bf16_wmma.hip",
            "232e0333d31c902729ae3f4e7174a2e253a75ba6ecfb229d3351d88649feafe6",
        ),
    ),
)
def test_representative_probe_source_hashes_are_unchanged(
    filename: str, expected_sha256: str
) -> None:
    source = resources.files("sol_execbench.data.hardware_calibration_probes").joinpath(
        filename
    )

    assert hashlib.sha256(source.read_bytes()).hexdigest() == expected_sha256


def test_production_probe_module_does_not_embed_hip_programs() -> None:
    source = Path(hip_probe_module.__file__).read_text(encoding="utf-8")

    assert "#include <hip/" not in source
    assert "__global__ void" not in source


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
        execute_candidate=cast(
            Callable[[CalibrationProfileKey], ProbeExecution | None],
            lambda _: object(),
        ),
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


def test_wmma_source_uses_fp16_wmma_intrinsic_when_requested() -> None:
    source = _hip_source(
        CalibrationProfileKey("compute", "matrix", "fp16", "fp16", "wmma")
    )

    assert "wmma_f32_16x16x16_f16_w32_gfx12" in source
    assert "0x3c00" in source


def test_fp16_stream_probe_uses_half_storage_and_byte_accounting() -> None:
    source = _hip_source(
        CalibrationProfileKey("memory", "stream_copy", "fp16", "fp16", "gfx12")
    )

    assert "__half" in source
    assert "sizeof(__half)" in source


def test_bf16_stream_probe_uses_bf16_storage_and_validation() -> None:
    source = _hip_source(
        CalibrationProfileKey("memory", "stream_copy", "bf16", "bf16", "gfx12")
    )

    assert "hip_bfloat16" in source
    assert "static_cast<float>" in source


def test_bf16_fp32_conversion_probe_accounts_for_both_storage_types() -> None:
    source = _hip_source(
        CalibrationProfileKey("memory", "stream_copy", "bf16", "fp32", "gfx12")
    )

    assert "conversion_probe_kernel" in source
    assert "sizeof(hip_bfloat16) + sizeof(float)" in source


def test_reduction_and_transcendental_profiles_have_dedicated_probes() -> None:
    reduction = _hip_source(
        CalibrationProfileKey("compute", "reduction", "fp32", "fp32", "gfx12")
    )
    transcendental = _hip_source(
        CalibrationProfileKey("compute", "transcendental", "fp32", "fp32", "gfx12")
    )

    assert "reduction_probe_kernel" in reduction
    assert "tanhf" in transcendental


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
        "hipLaunchKernelGGL(matrix_probe_kernel, dim3(grid_blocks), dim3(lane_count), 0, 0, device_output);"
    )
    validation_sync = source.index("hipDeviceSynchronize()", validation_launch)
    validation_copy = source.index(
        "hipMemcpy(output.data(), device_output", validation_sync
    )
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
    assert "const int blocks_per_cu = 8" in source
    assert "properties.multiProcessorCount" in source
    assert validation_failure < result_output


def test_matrix_compile_targets_selected_architecture(tmp_path) -> None:
    commands: list[list[str]] = []
    backend = HipCommandBackend(
        workspace=tmp_path,
        hipcc="hipcc",
        architecture="gfx1200",
        run=lambda command, **_: (
            commands.append(command) or type("Result", (), {"returncode": 0})()
        ),
    )

    assert (
        backend.compile(
            CalibrationProfileKey("compute", "matrix", "bf16", "bf16", "wmma")
        )
        == "passed"
    )
    assert "--offload-arch=gfx1200" in commands[0]


def test_extended_profiles_compile_when_declared(tmp_path) -> None:
    commands: list[list[str]] = []
    backend = HipCommandBackend(
        workspace=tmp_path,
        hipcc="hipcc",
        architecture="gfx1200",
        run=lambda command, **_: (
            commands.append(command) or type("Result", (), {"returncode": 0})()
        ),
    )

    for key in (
        CalibrationProfileKey("compute", "matrix", "fp32", "fp32", "gfx12"),
        CalibrationProfileKey("compute", "vector", "bf16", "bf16", "gfx12"),
        CalibrationProfileKey("memory", "stream_copy", "bf16", "bf16", "gfx12"),
        CalibrationProfileKey("compute", "reduction", "fp32", "fp32", "gfx12"),
        CalibrationProfileKey("compute", "transcendental", "fp32", "fp32", "gfx12"),
    ):
        assert backend.compile(key) == "passed"

    assert all("--offload-arch=gfx1200" in command for command in commands)


def test_fp32_vector_probe_uses_arithmetic_saturation_and_full_validation() -> None:
    source = _hip_source(
        CalibrationProfileKey("compute", "vector", "fp32", "fp32", "portable")
    )

    assert "arithmetic_repetitions = 4096" in source
    assert "fmaf(value" in source
    assert "2.0 * arithmetic_repetitions" in source
    assert "for (size_t index = 0; index < count; ++index) if" in source


def test_streaming_probe_requires_a_cache_busting_vram_working_set() -> None:
    source = _hip_source(
        CalibrationProfileKey("memory", "stream_copy", "fp32", "fp32", "portable")
    )

    assert "hipMemGetInfo" in source
    assert "maximum_count = static_cast<size_t>(1) << 27" in source
    assert "count < (static_cast<size_t>(1) << 24)" in source
    assert "2.0 * sizeof(float)" in source
