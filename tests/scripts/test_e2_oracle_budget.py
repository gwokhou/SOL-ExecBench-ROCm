"""Check cumulative CPU accounting and signal cleanup on the oracle boundary."""

import signal

import pytest

from scripts.internal.solar import e2_oracle_budget
from scripts.internal.solar.e2_oracle_budget import OracleCpuBudget


def test_failed_oracle_call_is_charged_and_restores_timer(monkeypatch):
    ticks = iter((10.0, 10.25))
    monkeypatch.setattr(
        e2_oracle_budget.time, "process_time", lambda: next(ticks)
    )
    prior_handler = signal.getsignal(signal.SIGPROF)
    prior_timer = signal.getitimer(signal.ITIMER_PROF)
    budget = OracleCpuBudget(2.0)
    with pytest.raises(ValueError, match="uncertain"), budget.charge():
        raise ValueError("uncertain")
    assert budget.spent_seconds == 0.25
    assert signal.getsignal(signal.SIGPROF) == prior_handler
    assert signal.getitimer(signal.ITIMER_PROF) == prior_timer


def test_exhausted_oracle_allowance_cannot_launch_a_further_call(monkeypatch):
    ticks = iter((0.0, 1.125))
    monkeypatch.setattr(
        e2_oracle_budget.time, "process_time", lambda: next(ticks)
    )
    budget = OracleCpuBudget(1.0)
    with (
        pytest.raises(TimeoutError, match="CPU allowance exhausted"),
        budget.charge(),
    ):
        pass
    assert budget.spent_seconds == 1.125
    with (
        pytest.raises(TimeoutError, match="CPU allowance exhausted"),
        budget.charge(),
    ):
        pytest.fail("the next oracle call ran after exhaustion")
