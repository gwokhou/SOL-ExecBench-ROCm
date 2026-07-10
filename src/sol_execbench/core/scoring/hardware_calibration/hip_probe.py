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
        if result == "unsupported":
            return "unsupported"
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
        hipcc: str | None = None,
        run: Callable[..., object] = subprocess.run,
    ) -> None:
        self.workspace = workspace or Path(mkdtemp(prefix="sol-execbench-hip-"))
        self.hipcc = hipcc if hipcc is not None else shutil.which("hipcc")
        self.run = run
        self._executables: dict[CalibrationProfileKey, Path] = {}

    def compile(self, key: CalibrationProfileKey) -> str:
        if self.hipcc is None:
            return "missing"
        self.workspace.mkdir(parents=True, exist_ok=True)
        stem = key.value.replace(".", "_")
        source = self.workspace / f"{stem}.hip"
        executable = self.workspace / stem
        try:
            source.write_text(_hip_source(key), encoding="utf-8")
            result = self.run(
                [self.hipcc, "-O3", str(source), "-o", str(executable)],
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
        if not samples or any(sample <= 0.0 for sample in samples):
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


def default_hip_probe(backend: HipCommandBackend | None = None) -> HipProbe:
    """Return the concrete HIP probe without resolving HIP tools at import time."""
    backend = backend or HipCommandBackend()
    return HipProbe(
        compile_candidate=backend.compile,
        execute_candidate=backend.execute,
        check_correctness=backend.correctness,
        check_stability=backend.stability,
    )


def _hip_source(key: CalibrationProfileKey) -> str:
    """Generate a self-checking FP32 vector or streaming-copy HIP benchmark."""
    is_compute = key.kind == "compute"
    kernel = (
        "out[index] = left[index] + right[index];"
        if is_compute
        else "out[index] = left[index];"
    )
    expected = "3.0f" if is_compute else "static_cast<float>(index)"
    rate = (
        "(static_cast<double>(count) / elapsed_ms / 1.0e9)"
        if is_compute
        else "(static_cast<double>(count) * 2.0 * sizeof(float) / elapsed_ms / 1.0e6)"
    )
    return f"""#include <hip/hip_runtime.h>
#include <cstdio>
#include <vector>

__global__ void probe_kernel(const float* left, const float* right, float* out, size_t count) {{
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) {{ {kernel} }}
}}

int main() {{
  constexpr size_t count = 1 << 20;
  std::vector<float> left(count), right(count), out(count);
  for (size_t index = 0; index < count; ++index) {{ left[index] = static_cast<float>(index); right[index] = 3.0f - left[index]; }}
  float *d_left = nullptr, *d_right = nullptr, *d_out = nullptr;
  if (hipMalloc(&d_left, count * sizeof(float)) || hipMalloc(&d_right, count * sizeof(float)) || hipMalloc(&d_out, count * sizeof(float))) return 2;
  hipMemcpy(d_left, left.data(), count * sizeof(float), hipMemcpyHostToDevice);
  hipMemcpy(d_right, right.data(), count * sizeof(float), hipMemcpyHostToDevice);
  hipEvent_t start, stop; hipEventCreate(&start); hipEventCreate(&stop);
  for (int repeat = 0; repeat < 7; ++repeat) {{
    hipEventRecord(start); hipLaunchKernelGGL(probe_kernel, dim3((count + 255) / 256), dim3(256), 0, 0, d_left, d_right, d_out, count); hipEventRecord(stop); hipEventSynchronize(stop);
    float elapsed_ms = 0.0f; hipEventElapsedTime(&elapsed_ms, start, stop);
    if (elapsed_ms <= 0.0f) return 3;
    std::printf("RESULT %.9f\\n", {rate});
  }}
  hipMemcpy(out.data(), d_out, count * sizeof(float), hipMemcpyDeviceToHost);
  for (size_t index = 0; index < count; ++index) if (out[index] != {expected}) return 4;
  hipFree(d_left); hipFree(d_right); hipFree(d_out); hipEventDestroy(start); hipEventDestroy(stop); return 0;
}}
"""
