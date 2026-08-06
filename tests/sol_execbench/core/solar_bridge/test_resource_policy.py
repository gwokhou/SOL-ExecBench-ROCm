from __future__ import annotations

from sol_execbench.core.solar_bridge.resource_policy import (
    formal_mapper_thread_count,
)
from solar.analysis.orojenesis.configuration import OROJENESIS_MAPPER_THREADS


def test_formal_mapper_thread_count_uses_canonical_solar_policy() -> None:
    assert formal_mapper_thread_count() == OROJENESIS_MAPPER_THREADS == 8
