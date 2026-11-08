"""Optional process-local diagnostic observation; never changes result schemas."""

import resource
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from enum import StrEnum
from functools import wraps

from solar.contracts import SolarStage
from solar.memory_observation import StageMemoryObservation


class ObservationStage(StrEnum):
    """Auxiliary operations outside the public pipeline stage vocabulary."""

    MAPPER = "mapper"


_SINK: ContextVar[list[dict[str, object]] | None] = ContextVar(
    "solar_stage_cost_sink", default=None
)


@contextmanager
def collect_stage_costs(records: list[dict[str, object]]) -> Iterator[None]:
    """Enable auxiliary costs only for the caller's diagnostic activity."""
    token = _SINK.set(records)
    try:
        yield
    finally:
        _SINK.reset(token)


def observed_stage[**P, R](
    stage: SolarStage | ObservationStage,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Measure actual stage wall time and process RSS high water, even on failure."""

    def decorate(function: Callable[P, R]) -> Callable[P, R]:
        @wraps(function)
        def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
            sink = _SINK.get()
            if sink is None:
                return function(*args, **kwargs)
            memory = StageMemoryObservation(
                observe_device=stage
                in {
                    SolarStage.GRAPH_EXTRACTION,
                    SolarStage.IR_CONVERSION,
                    SolarStage.CONVERSION_VERIFICATION,
                }
            )
            memory.start()
            start = time.monotonic()
            try:
                return function(*args, **kwargs)
            finally:
                sink.append(
                    {
                        "stage": stage.value,
                        "wall_seconds": time.monotonic() - start,
                        "host_process_high_water_bytes": resource.getrusage(
                            resource.RUSAGE_SELF
                        ).ru_maxrss
                        * 1024,
                        **memory.finish(),
                    }
                )

        return wrapped

    return decorate
