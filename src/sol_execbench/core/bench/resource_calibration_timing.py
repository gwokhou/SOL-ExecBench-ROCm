"""Wall-time observations for resource calibration; never qualification authority."""

from __future__ import annotations

import json
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import StrEnum
from time import perf_counter


class CalibrationPhase(StrEnum):
    """Non-overlapping top-level phases in one calibration invocation."""

    PREPARE = "cpu_prepare"
    GPU_LIFECYCLE = "gpu_lifecycle_including_wait_and_cleanup"
    FINALIZE = "cpu_finalize"


@dataclass(slots=True, kw_only=True)
class CalibrationTimings:
    """Emit bounded phase durations even when a phase raises or is cancelled."""

    seconds: dict[CalibrationPhase, float] = field(default_factory=dict)
    completed: dict[CalibrationPhase, bool] = field(default_factory=dict)

    @contextmanager
    def measure(self, phase: CalibrationPhase) -> Iterator[None]:
        """Time one phase, retaining failure and cancellation observations."""
        start = perf_counter()
        self.completed[phase] = False
        try:
            yield
            self.completed[phase] = True
        finally:
            self.seconds[phase] = perf_counter() - start

    def report(self) -> None:
        """Leave stdout's existing gate/result protocol untouched."""
        print(
            "calibration_timing "
            + json.dumps(
                {"seconds": self.seconds, "completed": self.completed}
            ),
            file=sys.stderr,
        )
