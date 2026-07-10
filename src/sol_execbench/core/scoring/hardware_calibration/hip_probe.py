# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Small, injectable HIP calibration probe abstraction.

The production implementation deliberately has no import-time HIP dependency.
Callers provide compilation and execution seams, which keeps calibration tests
portable and prevents a missing ROCm installation from becoming fabricated data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from sol_execbench.core.scoring.hardware_calibration.models import CalibrationCandidate


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


CompileCandidate = Callable[[CalibrationProfileKey], str | bool | None]
ExecuteCandidate = Callable[[CalibrationProfileKey], ProbeExecution | None]
CheckCandidate = Callable[[CalibrationProfileKey, ProbeExecution], bool | None]


@dataclass(frozen=True)
class HipProbe:
    """Compile, run, check, and stabilise a HIP candidate through injected seams."""

    compile_candidate: CompileCandidate | None
    execute_candidate: ExecuteCandidate | None
    check_correctness: CheckCandidate | None
    check_stability: CheckCandidate | None

    def measure(self, key: CalibrationProfileKey) -> CalibrationCandidate:
        compile_state = self._compile(key)
        if compile_state == "unsupported":
            return self._unavailable(key, "hip_probe_explicitly_unsupported")
        if compile_state != "passed":
            return self._unknown(key, f"hip_probe_compile_{compile_state}")
        execution = self._execute(key)
        if execution is None:
            return self._unknown(key, "hip_probe_run_failed")
        if not self._check(self.check_correctness, key, execution):
            return self._unknown(key, "hip_probe_correctness_failed")
        if not self._check(self.check_stability, key, execution):
            return self._unknown(key, "hip_probe_stability_failed")
        try:
            return CalibrationCandidate(
                key=key.value,
                state="measured",
                value=min(execution.samples),
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
