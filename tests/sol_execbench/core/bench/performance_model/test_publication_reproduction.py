from __future__ import annotations

from sol_execbench.core.bench.performance_model import publication


def test_numeric_reproduction_allows_only_bounded_float_drift() -> None:
    expected = {
        "identity": "fixed",
        "coefficients": [1.0, 1e-12],
        "nested": {"threshold": 2.0},
    }

    assert publication._numerically_equivalent(
        expected,
        {
            "identity": "fixed",
            "coefficients": [1.0 + 5e-10, 2e-12],
            "nested": {"threshold": 2.0 - 5e-10},
        },
    )
    assert not publication._numerically_equivalent(
        expected,
        {
            "identity": "fixed",
            "coefficients": [1.0 + 5e-8, 1e-12],
            "nested": {"threshold": 2.0},
        },
    )


def test_numeric_reproduction_keeps_discrete_fields_exact() -> None:
    assert not publication._numerically_equivalent(
        {"identity": "fixed", "enabled": True},
        {"identity": "changed", "enabled": True},
    )
    assert not publication._numerically_equivalent(
        {"identity": "fixed", "enabled": True},
        {"identity": "fixed", "enabled": False},
    )
