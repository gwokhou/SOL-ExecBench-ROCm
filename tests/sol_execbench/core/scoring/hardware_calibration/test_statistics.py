from sol_execbench.core.scoring.hardware_calibration.statistics import (
    select_conservative_value,
)


def test_select_conservative_value_uses_minimum_of_best_three() -> None:
    result = select_conservative_value((1.0, 4.9, 5.0, 2.0, 5.1, 5.05, 3.0))

    assert result.value == 5.0
    assert result.retained_samples == (5.1, 5.05, 5.0)
