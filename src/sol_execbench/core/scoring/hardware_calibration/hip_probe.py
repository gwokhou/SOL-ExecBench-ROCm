# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Small, injectable HIP calibration probe abstraction.

The production implementation deliberately has no import-time HIP dependency.
Callers provide compilation and execution seams, which keeps calibration tests
portable and prevents a missing ROCm installation from becoming fabricated data.
"""

from __future__ import annotations

import hashlib
import subprocess
from importlib import resources
from math import isfinite
from pathlib import Path
from dataclasses import dataclass
from tempfile import mkdtemp
from typing import Any, Callable, Mapping

from sol_execbench.core.scoring.hardware_calibration.models import CalibrationCandidate
from sol_execbench.core.platform.runtime import resolve_rocm_tool
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
ProbeEvidence = Callable[[CalibrationProfileKey], Mapping[str, Any] | None]
_DEFAULT_HIPCC = object()
_PROBE_RESOURCE_PACKAGE = "sol_execbench.data.hardware_calibration_probes"
_PROBE_RESOURCES: dict[tuple[str, str, str, str | None], str] = {
    ("vector", "fp32", "fp32", None): "vector_fp32_fp32.hip",
    ("stream_copy", "fp32", "fp32", None): "stream_copy_fp32_fp32.hip",
    ("vector", "fp16", "fp16", None): "vector_fp16_fp16.hip",
    ("stream_copy", "fp16", "fp16", None): "stream_copy_fp16_fp16.hip",
    ("vector", "bf16", "bf16", None): "vector_bf16_bf16.hip",
    ("stream_copy", "bf16", "bf16", None): "stream_copy_bf16_bf16.hip",
    ("stream_copy", "bf16", "fp32", None): "stream_copy_bf16_fp32.hip",
    ("stream_copy", "fp32", "bf16", None): "stream_copy_fp32_bf16.hip",
    ("reduction", "fp32", "fp32", None): "reduction_fp32_fp32.hip",
    ("transcendental", "fp32", "fp32", None): "transcendental_fp32_fp32.hip",
    # gfx12 FP32 matrix math uses the ordinary FP32 FMA datapath rather than
    # WMMA. Reuse the self-checking saturation probe for that shared resource.
    ("matrix", "fp32", "fp32", "gfx12"): "vector_fp32_fp32.hip",
    ("matrix", "bf16", "bf16", "wmma"): "matrix_bf16_bf16_wmma.hip",
    ("matrix", "fp16", "fp16", "wmma"): "matrix_fp16_fp16_wmma.hip",
    ("matrix", "bf16", "bf16", "mfma"): "matrix_bf16_bf16_mfma.hip",
}


@dataclass(frozen=True)
class HipProbe:
    """Compile, run, check, and stabilise a HIP candidate through injected seams."""

    compile_candidate: CompileCandidate | None = None
    execute_candidate: ExecuteCandidate | None = None
    check_correctness: CheckCandidate | None = None
    check_stability: CheckCandidate | None = None
    evidence_for: ProbeEvidence | None = None

    def provenance_for(self, key: CalibrationProfileKey) -> dict[str, Any]:
        """Return compile/run provenance when the concrete backend exposes it."""
        if self.evidence_for is None:
            return {"collection": "injected_probe"}
        try:
            evidence = self.evidence_for(key)
        except Exception:
            return {"collection": "unavailable"}
        return (
            dict(evidence)
            if isinstance(evidence, Mapping)
            else {"collection": "unavailable"}
        )

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
        if hipcc is _DEFAULT_HIPCC:
            discovered_hipcc = resolve_rocm_tool("hipcc")
            self.hipcc = str(discovered_hipcc) if discovered_hipcc else None
        else:
            self.hipcc = hipcc
        self.architecture = architecture.lower() if architecture is not None else None
        self.run = run
        self._executables: dict[CalibrationProfileKey, Path] = {}
        self._evidence: dict[CalibrationProfileKey, dict[str, Any]] = {}

    def compile(self, key: CalibrationProfileKey) -> str:
        if self.architecture is not None and not _matrix_path_is_supported(
            key, self.architecture
        ):
            return "unsupported"
        if self.hipcc is None:
            return "missing"
        is_matrix = (
            key.operation == "matrix"
            and key.input_dtype in {"bf16", "fp16", "fp32"}
            and key.output_dtype == key.input_dtype
            and key.path in {"wmma", "mfma", "gfx12"}
        )
        is_vector_probe = (
            key.input_dtype in {"fp32", "fp16", "bf16"}
            and key.output_dtype == key.input_dtype
            and key.operation == "vector"
        )
        is_stream_probe = key.operation == "stream_copy" and (
            key.input_dtype,
            key.output_dtype,
        ) in {
            ("fp32", "fp32"),
            ("fp16", "fp16"),
            ("bf16", "bf16"),
            ("bf16", "fp32"),
            ("fp32", "bf16"),
        }
        is_reduction_probe = (
            key.kind == "compute"
            and key.operation == "reduction"
            and key.input_dtype == key.output_dtype == "fp32"
        )
        is_transcendental_probe = (
            key.kind == "compute"
            and key.operation == "transcendental"
            and key.input_dtype == key.output_dtype == "fp32"
        )
        if not (
            is_matrix
            or is_vector_probe
            or is_stream_probe
            or is_reduction_probe
            or is_transcendental_probe
        ):
            return "failed"
        self.workspace.mkdir(parents=True, exist_ok=True)
        stem = key.value.replace(".", "_")
        source = self.workspace / f"{stem}.hip"
        executable = self.workspace / stem
        try:
            source_text = _hip_source(key)
            source.write_text(source_text, encoding="utf-8")
            command = [self.hipcc, "-O3"]
            if self.architecture is not None:
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
        try:
            binary_sha256 = hashlib.sha256(executable.read_bytes()).hexdigest()
        except OSError:
            binary_sha256 = None
        self._evidence[key] = {
            "compiler_command": command,
            "source_sha256": hashlib.sha256(source_text.encode("utf-8")).hexdigest(),
            "binary_sha256": binary_sha256,
            "warmup_iterations": 3,
            "timed_samples": 7,
        }
        return "passed"

    def evidence(self, key: CalibrationProfileKey) -> Mapping[str, Any] | None:
        return self._evidence.get(key)

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
        evidence_for=backend.evidence,
    )


def _matrix_path_is_supported(key: CalibrationProfileKey, architecture: str) -> bool:
    """Return whether a declared matrix path matches an AMD ISA family."""
    if key.operation != "matrix":
        return True
    if key.output_dtype != key.input_dtype:
        return False
    if key.input_dtype == "fp32":
        return architecture.startswith("gfx12") and key.path == "gfx12"
    if key.input_dtype == "fp16":
        return architecture.startswith("gfx12") and key.path == "wmma"
    if key.input_dtype == "bf16":
        return (architecture.startswith("gfx12") and key.path == "wmma") or (
            architecture.startswith(("gfx94", "gfx95")) and key.path == "mfma"
        )
    return False


def _hip_source(key: CalibrationProfileKey) -> str:
    """Load the self-checking HIP benchmark for a calibration candidate."""
    route = (
        key.operation,
        key.input_dtype,
        key.output_dtype,
        key.path if key.operation == "matrix" else None,
    )
    try:
        filename = _PROBE_RESOURCES[route]
    except KeyError as exc:
        raise ValueError(f"unsupported HIP calibration probe: {key.value}") from exc
    return _read_probe_source(filename)


def _read_probe_source(filename: str) -> str:
    """Read one UTF-8 HIP package resource with a consistent failure mode."""
    resource = resources.files(_PROBE_RESOURCE_PACKAGE).joinpath(filename)
    if not resource.is_file():
        raise RuntimeError(f"missing HIP calibration probe resource: {filename}")
    try:
        return resource.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise RuntimeError(
            f"cannot read HIP calibration probe resource: {filename}"
        ) from exc
