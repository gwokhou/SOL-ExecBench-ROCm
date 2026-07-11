# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Small, injectable HIP calibration probe abstraction.

The production implementation deliberately has no import-time HIP dependency.
Callers provide compilation and execution seams, which keeps calibration tests
portable and prevents a missing ROCm installation from becoming fabricated data.
"""

from __future__ import annotations

import shutil
import subprocess
from math import isfinite
from pathlib import Path
from tempfile import mkdtemp
from dataclasses import dataclass
from typing import Callable

from sol_execbench.core.scoring.hardware_calibration.models import CalibrationCandidate
from sol_execbench.core.scoring.hardware_calibration.statistics import (
    MINIMUM_SAMPLE_COUNT,
    select_conservative_value,
)


@dataclass(frozen=True, order=True)
class CalibrationProfileKey:
    """Identifies one measured calibration matrix entry."""

    kind: str
    operation: str
    input_dtype: str
    output_dtype: str
    path: str

    @property
    def value(self) -> str:
        return ".".join(
            (self.kind, self.operation, self.input_dtype, self.output_dtype, self.path)
        )


@dataclass(frozen=True)
class ProbeExecution:
    """Timed samples produced by a successfully executed probe."""

    samples: tuple[float, ...]
    unit: str
    correct: bool = True
    stable: bool = True


CompileCandidate = Callable[[CalibrationProfileKey], str | bool | None]
ExecuteCandidate = Callable[[CalibrationProfileKey], ProbeExecution | None]
CheckCandidate = Callable[[CalibrationProfileKey, ProbeExecution], bool | None]
_DEFAULT_HIPCC = object()


@dataclass(frozen=True)
class HipProbe:
    """Compile, run, check, and stabilise a HIP candidate through injected seams."""

    compile_candidate: CompileCandidate | None = None
    execute_candidate: ExecuteCandidate | None = None
    check_correctness: CheckCandidate | None = None
    check_stability: CheckCandidate | None = None

    def measure(self, key: CalibrationProfileKey) -> CalibrationCandidate:
        compile_state = self._compile(key)
        if compile_state == "unsupported":
            return self._unavailable(key, "hip_probe_explicitly_unsupported")
        if compile_state != "passed":
            return self._unknown(key, f"hip_probe_compile_{compile_state}")
        execution = self._execute(key)
        if execution is None:
            return self._unknown(key, "hip_probe_run_failed")
        if not isinstance(execution, ProbeExecution):
            return self._unknown(key, "hip_probe_execution_malformed")
        if not self._check(self.check_correctness, key, execution):
            return self._unknown(key, "hip_probe_correctness_failed")
        if not self._check(self.check_stability, key, execution):
            return self._unknown(key, "hip_probe_stability_failed")
        try:
            return CalibrationCandidate(
                key=key.value,
                state="measured",
                value=select_conservative_value(execution.samples).value,
                unit=execution.unit,
                samples=execution.samples,
            )
        except (TypeError, ValueError):
            return self._unknown(key, "hip_probe_samples_invalid")

    def _compile(self, key: CalibrationProfileKey) -> str:
        if self.compile_candidate is None:
            return "missing"
        try:
            result = self.compile_candidate(key)
        except Exception:
            return "failed"
        if result in {"unsupported", "missing", "failed"}:
            return result
        return "passed" if result is True or result == "passed" else "failed"

    def _execute(self, key: CalibrationProfileKey) -> ProbeExecution | None:
        if self.execute_candidate is None:
            return None
        try:
            return self.execute_candidate(key)
        except Exception:
            return None

    @staticmethod
    def _check(
        check: CheckCandidate | None,
        key: CalibrationProfileKey,
        execution: ProbeExecution,
    ) -> bool:
        if check is None:
            return False
        try:
            return check(key, execution) is True
        except Exception:
            return False

    @staticmethod
    def _unknown(key: CalibrationProfileKey, reason_code: str) -> CalibrationCandidate:
        return CalibrationCandidate(
            key.value, "unknown", None, None, reason_code=reason_code
        )

    @staticmethod
    def _unavailable(
        key: CalibrationProfileKey, reason_code: str
    ) -> CalibrationCandidate:
        return CalibrationCandidate(
            key.value, "unavailable", None, None, reason_code=reason_code
        )


class HipCommandBackend:
    """Concrete HIPCC backend, lazy enough to remain usable off ROCm hosts."""

    def __init__(
        self,
        workspace: Path | None = None,
        hipcc: str | None | object = _DEFAULT_HIPCC,
        architecture: str | None = None,
        run: Callable[..., object] = subprocess.run,
    ) -> None:
        self.workspace = workspace or Path(mkdtemp(prefix="sol-execbench-hip-"))
        self.hipcc = shutil.which("hipcc") if hipcc is _DEFAULT_HIPCC else hipcc
        self.architecture = architecture.lower() if architecture is not None else None
        self.run = run
        self._executables: dict[CalibrationProfileKey, Path] = {}

    def compile(self, key: CalibrationProfileKey) -> str:
        if self.architecture is not None and not _matrix_path_is_supported(
            key, self.architecture
        ):
            return "unsupported"
        if self.hipcc is None:
            return "missing"
        is_matrix = (
            key.operation == "matrix"
            and key.input_dtype in {"bf16", "fp16"}
            and key.output_dtype == key.input_dtype
            and key.path in {"wmma", "mfma"}
        )
        is_fp32_probe = (
            key.input_dtype == "fp32"
            and key.output_dtype == "fp32"
            and key.operation == "vector"
        )
        is_stream_probe = (
            key.input_dtype in {"fp32", "fp16"}
            and key.output_dtype == key.input_dtype
            and key.operation == "stream_copy"
        )
        if not (is_matrix or is_fp32_probe or is_stream_probe):
            return "failed"
        self.workspace.mkdir(parents=True, exist_ok=True)
        stem = key.value.replace(".", "_")
        source = self.workspace / f"{stem}.hip"
        executable = self.workspace / stem
        try:
            source.write_text(_hip_source(key), encoding="utf-8")
            command = [self.hipcc, "-O3"]
            if is_matrix and self.architecture is not None:
                command.append(f"--offload-arch={self.architecture}")
            command.extend((str(source), "-o", str(executable)))
            result = self.run(
                command,
                capture_output=True,
                text=True,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return "failed"
        if getattr(result, "returncode", 1) != 0:
            return "failed"
        self._executables[key] = executable
        return "passed"

    def execute(self, key: CalibrationProfileKey) -> ProbeExecution | None:
        executable = self._executables.get(key)
        if executable is None:
            return None
        try:
            result = self.run(
                [str(executable)], capture_output=True, text=True, check=False
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if getattr(result, "returncode", 1) != 0:
            return None
        try:
            samples = tuple(
                float(line.split()[1])
                for line in str(getattr(result, "stdout", "")).splitlines()
                if line.startswith("RESULT ")
            )
        except (IndexError, ValueError):
            return None
        if not samples or any(
            not isfinite(sample) or sample <= 0.0 for sample in samples
        ):
            return None
        return ProbeExecution(
            samples=samples,
            unit="TFLOP/s" if key.kind == "compute" else "GB/s",
        )

    @staticmethod
    def correctness(_: CalibrationProfileKey, execution: ProbeExecution) -> bool:
        return execution.correct

    @staticmethod
    def stability(_: CalibrationProfileKey, execution: ProbeExecution) -> bool:
        return execution.stable and len(execution.samples) >= MINIMUM_SAMPLE_COUNT


def default_hip_probe(
    backend: HipCommandBackend | None = None, architecture: str | None = None
) -> HipProbe:
    """Return the concrete HIP probe without resolving HIP tools at import time."""
    backend = backend or HipCommandBackend(architecture=architecture)
    return HipProbe(
        compile_candidate=backend.compile,
        execute_candidate=backend.execute,
        check_correctness=backend.correctness,
        check_stability=backend.stability,
    )


def _matrix_path_is_supported(key: CalibrationProfileKey, architecture: str) -> bool:
    """Return whether a declared matrix path matches an AMD ISA family."""
    if (
        key.operation != "matrix"
        or key.input_dtype not in {"bf16", "fp16"}
        or key.output_dtype != key.input_dtype
    ):
        return True
    if key.input_dtype == "fp16":
        return architecture.startswith("gfx12") and key.path == "wmma"
    return (architecture.startswith("gfx12") and key.path == "wmma") or (
        architecture.startswith(("gfx94", "gfx95")) and key.path == "mfma"
    )


def _hip_source(key: CalibrationProfileKey) -> str:
    """Generate the self-checking HIP benchmark for a calibration candidate."""
    if (
        key.operation == "matrix"
        and key.input_dtype in {"bf16", "fp16"}
        and key.output_dtype == key.input_dtype
        and key.path in {"wmma", "mfma"}
    ):
        return _matrix_hip_source(key)

    is_compute = key.kind == "compute"
    scalar_type = "__half" if key.input_dtype == "fp16" else "float"
    if is_compute:
        kernel = """float value = left[index];
    #pragma unroll 16
    for (int repeat = 0; repeat < arithmetic_repetitions; ++repeat) {
      value = fmaf(value, right[index], 0.0000001f);
    }
    out[index] = value;"""
        count_declaration = "constexpr size_t count = 1 << 20;"
        initialization = "left[index] = 1.0f; right[index] = 1.0000001f;"
        expected_setup = """float expected = 1.0f;
  for (int repeat = 0; repeat < arithmetic_repetitions; ++repeat) expected = fmaf(expected, 1.0000001f, 0.0000001f);"""
        expected = "expected"
        rate = "(static_cast<double>(count) * 2.0 * arithmetic_repetitions / elapsed_ms / 1.0e9)"
    else:
        kernel = "out[index] = left[index];"
        # Size the streaming set from available VRAM and keep a 512 MiB cap per
        # buffer.  Three device buffers remain well beyond gfx1200 cache sizes.
        count_declaration = f"""size_t free_bytes = 0, total_bytes = 0;
  if (hipMemGetInfo(&free_bytes, &total_bytes) != hipSuccess) return 2;
  const size_t maximum_count = static_cast<size_t>(1) << 27;
  const size_t count = std::min(maximum_count, free_bytes / (sizeof({scalar_type}) * 12));
  if (count < (static_cast<size_t>(1) << 24)) return 5;"""
        initialization = f"left[index] = static_cast<{scalar_type}>(1.0f); right[index] = static_cast<{scalar_type}>(0.0f);"
        expected_setup = ""
        expected = "1.0f"
        rate = f"(static_cast<double>(count) * 2.0 * sizeof({scalar_type}) / elapsed_ms / 1.0e6)"
    return f"""#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

constexpr int arithmetic_repetitions = 4096;

__global__ void probe_kernel(const {scalar_type}* left, const {scalar_type}* right, {scalar_type}* out, size_t count) {{
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) {{ {kernel} }}
}}

int main() {{
  {count_declaration}
  std::vector<{scalar_type}> left(count), right(count), out(count);
  for (size_t index = 0; index < count; ++index) {{ {initialization} }}
  {scalar_type} *d_left = nullptr, *d_right = nullptr, *d_out = nullptr;
  if (hipMalloc(&d_left, count * sizeof({scalar_type})) || hipMalloc(&d_right, count * sizeof({scalar_type})) || hipMalloc(&d_out, count * sizeof({scalar_type}))) return 2;
  hipMemcpy(d_left, left.data(), count * sizeof({scalar_type}), hipMemcpyHostToDevice);
  hipMemcpy(d_right, right.data(), count * sizeof({scalar_type}), hipMemcpyHostToDevice);
  hipEvent_t start, stop; hipEventCreate(&start); hipEventCreate(&stop);
  for (int warmup = 0; warmup < 3; ++warmup) {{
    hipLaunchKernelGGL(probe_kernel, dim3((count + 255) / 256), dim3(256), 0, 0, d_left, d_right, d_out, count);
  }}
  hipDeviceSynchronize();
  for (int repeat = 0; repeat < 7; ++repeat) {{
    hipEventRecord(start); hipLaunchKernelGGL(probe_kernel, dim3((count + 255) / 256), dim3(256), 0, 0, d_left, d_right, d_out, count); hipEventRecord(stop); hipEventSynchronize(stop);
    float elapsed_ms = 0.0f; hipEventElapsedTime(&elapsed_ms, start, stop);
    if (elapsed_ms <= 0.0f) return 3;
    std::printf("RESULT %.9f\\n", {rate});
  }}
  hipMemcpy(out.data(), d_out, count * sizeof({scalar_type}), hipMemcpyDeviceToHost);
  {expected_setup}
  for (size_t index = 0; index < count; ++index) if (std::fabs(static_cast<float>(out[index]) - {expected}) > 1.0e-5f) return 4;
  hipFree(d_left); hipFree(d_right); hipFree(d_out); hipEventDestroy(start); hipEventDestroy(stop); return 0;
}}
"""


def _matrix_hip_source(key: CalibrationProfileKey) -> str:
    """Generate a direct-intrinsic matrix probe for the selected ISA path."""
    if key.path == "wmma":
        # gfx12 WMMA executes with wave32.  Every lane owns eight FP32 outputs.
        lane_count = 32
        values_per_lane = 8
        packed_bf16_count = 8
        accumulator_type = "float8v"
        intrinsic = (
            "__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, accumulator)"
            if key.input_dtype == "fp16"
            else "__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, b, accumulator)"
        )
    elif key.path == "mfma":
        # gfx94x/gfx95x MFMA executes with wave64.  Every lane owns four outputs.
        lane_count = 64
        values_per_lane = 4
        packed_bf16_count = 4
        accumulator_type = "float4v"
        intrinsic = (
            "__builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a, b, accumulator, 0, 0, 0)"
        )
    else:  # Defensive guard for direct callers outside _hip_source.
        raise ValueError(f"unsupported matrix path: {key.path}")

    packed_value = "0x3c00" if key.input_dtype == "fp16" else "0x3f80"
    packed_values = ", ".join(packed_value for _ in range(packed_bf16_count))
    return f"""#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <cmath>
#include <cstdio>
#include <vector>

// HIP provides the __hip_bfloat16 type used by this matrix probe.
using bf16x{packed_bf16_count} = unsigned short __attribute__((ext_vector_type({packed_bf16_count})));
using {accumulator_type} = float __attribute__((ext_vector_type({values_per_lane})));

constexpr int matrix_m = 16;
constexpr int matrix_n = 16;
constexpr int matrix_k = 16;
constexpr int repetitions = 1024;
constexpr int lane_count = {lane_count};
constexpr int values_per_lane = {values_per_lane};
constexpr float validation_tolerance = 1.0e-3f;

__global__ void matrix_probe_kernel(float* out) {{
  bf16x{packed_bf16_count} a = {{{packed_values}}};
  bf16x{packed_bf16_count} b = {{{packed_values}}};
  {accumulator_type} accumulator = {{}};
  for (int repeat = 0; repeat < repetitions; ++repeat) {{
    accumulator = {intrinsic};
  }}
  const int lane = threadIdx.x;
  const size_t block_offset = static_cast<size_t>(blockIdx.x) * matrix_m * matrix_n;
  for (int index = 0; index < values_per_lane; ++index) {{
    out[block_offset + lane * values_per_lane + index] = accumulator[index];
  }}
}}

int main() {{
  hipDeviceProp_t properties{{}};
  if (hipGetDeviceProperties(&properties, 0) != hipSuccess || properties.multiProcessorCount <= 0) return 2;
  // One wave cannot measure device throughput.  Launch several independent
  // waves per CU so the scheduler has enough work to hide matrix-instruction
  // latency and the FLOP accounting covers every launched block.
  const int blocks_per_cu = 8;
  const int grid_blocks = properties.multiProcessorCount * blocks_per_cu;
  const size_t output_count = static_cast<size_t>(grid_blocks) * matrix_m * matrix_n;
  std::vector<float> output(output_count);
  float* device_output = nullptr;
  if (hipMalloc(&device_output, output_count * sizeof(float)) != hipSuccess) return 2;
  hipEvent_t start = nullptr, stop = nullptr;
  if (hipEventCreate(&start) != hipSuccess || hipEventCreate(&stop) != hipSuccess) return 2;

  hipLaunchKernelGGL(matrix_probe_kernel, dim3(grid_blocks), dim3(lane_count), 0, 0, device_output);
  if (hipDeviceSynchronize() != hipSuccess) return 2;
  if (hipMemcpy(output.data(), device_output, output_count * sizeof(float), hipMemcpyDeviceToHost) != hipSuccess) return 2;
  const float cpu_reference = static_cast<float>(matrix_k * repetitions);
  for (float value : output) {{
    if (std::fabs(value - cpu_reference) > validation_tolerance) return 4;
  }}

  double samples[7] = {{}};
  for (int sample = 0; sample < 7; ++sample) {{
    if (hipEventRecord(start) != hipSuccess) return 2;
    hipLaunchKernelGGL(matrix_probe_kernel, dim3(grid_blocks), dim3(lane_count), 0, 0, device_output);
    if (hipEventRecord(stop) != hipSuccess || hipEventSynchronize(stop) != hipSuccess) return 2;
    float elapsed_ms = 0.0f;
    if (hipEventElapsedTime(&elapsed_ms, start, stop) != hipSuccess || elapsed_ms <= 0.0f) return 3;
    const double elapsed_seconds = static_cast<double>(elapsed_ms) / 1.0e3;
    samples[sample] = static_cast<double>(grid_blocks) * 2.0 * matrix_m * matrix_n * matrix_k * repetitions / elapsed_seconds / 1.0e12;
  }}

  for (double sample : samples) std::printf("RESULT %.9f\\n", sample);
  hipEventDestroy(start); hipEventDestroy(stop); hipFree(device_output);
  return 0;
}}
"""
