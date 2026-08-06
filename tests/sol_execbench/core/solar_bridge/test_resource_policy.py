from __future__ import annotations

from sol_execbench.core.solar_bridge import resource_policy
from sol_execbench.core.solar_bridge.resource_policy import (
    formal_mapper_thread_count,
)


def test_formal_mapper_thread_count_uses_canonical_solar_policy(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        resource_policy,
        "orojenesis_mapper_thread_count",
        lambda: 12,
    )

    assert formal_mapper_thread_count() == 12


def test_available_mapper_cpus_use_canonical_solar_policy(monkeypatch) -> None:
    monkeypatch.setattr(
        resource_policy,
        "available_logical_cpu_count",
        lambda: 6,
    )

    assert resource_policy.available_formal_mapper_logical_cpu_count() == 6
