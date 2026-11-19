"""Track only CPU time charged to the independent E2 oracle."""

import math
import signal
import time
from collections.abc import Iterator
from contextlib import contextmanager
from types import FrameType

from scripts.internal.solar.e2_contracts import ORACLE_BUDGET_SECONDS

ORACLE_ALLOWANCE_FIELD = "oracle_cpu_allowance_seconds"


def _expired(_signal: int, _frame: FrameType | None) -> None:
    raise TimeoutError("E2 cumulative oracle CPU allowance exhausted")


class OracleCpuBudget:
    """Carry cumulative CPU allowance across calls in one trusted process."""

    def __init__(self, seconds: float) -> None:
        """Require an allowance within the preregistered activity limit."""
        if (
            not math.isfinite(seconds)
            or not 0 < seconds <= ORACLE_BUDGET_SECONDS
        ):
            raise ValueError("invalid E2 oracle CPU allowance")
        self._limit = seconds
        self.spent_seconds = 0.0

    @contextmanager
    def charge(self) -> Iterator[None]:
        """Arm a CPU timer around one oracle call and retain failed call usage."""
        remaining = self._limit - self.spent_seconds
        if remaining <= 0:
            _expired(signal.SIGPROF, None)
        previous_handler = signal.signal(signal.SIGPROF, _expired)
        previous_timer: tuple[float, float] | None = None
        started = time.process_time()
        try:
            previous_timer = signal.setitimer(signal.ITIMER_PROF, remaining)
            yield
        finally:
            try:
                signal.setitimer(signal.ITIMER_PROF, 0)
            finally:
                self.spent_seconds += time.process_time() - started
                signal.signal(signal.SIGPROF, previous_handler)
                if previous_timer is not None:
                    signal.setitimer(signal.ITIMER_PROF, *previous_timer)
        if self.spent_seconds >= self._limit:
            _expired(signal.SIGPROF, None)
