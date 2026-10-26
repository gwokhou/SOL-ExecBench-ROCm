"""Analysis deadlines respect the caller's signal ownership."""

import signal

import pytest

from solar.analysis.interrupts import analysis_deadline, cleanup_on_termination


@pytest.mark.requires_linux
def test_active_caller_timer_is_not_replaced():
    previous = signal.getsignal(signal.SIGALRM)
    signal.setitimer(signal.ITIMER_REAL, 30)
    try:
        with (
            pytest.raises(ValueError, match="active caller timer"),
            analysis_deadline(1),
        ):
            pytest.fail("active caller timer must reject analysis")
        assert signal.getitimer(signal.ITIMER_REAL)[0] > 0
        assert signal.getsignal(signal.SIGALRM) == previous
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


@pytest.mark.requires_linux
def test_analysis_contexts_restore_signal_handlers_after_invariant_error():
    alarm, term = (
        signal.getsignal(signal.SIGALRM),
        signal.getsignal(signal.SIGTERM),
    )
    with (
        pytest.raises(RuntimeError, match="invariant"),
        cleanup_on_termination(),
        analysis_deadline(1),
    ):
        raise RuntimeError("invariant")
    assert signal.getsignal(signal.SIGALRM) == alarm
    assert signal.getsignal(signal.SIGTERM) == term
    assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)
