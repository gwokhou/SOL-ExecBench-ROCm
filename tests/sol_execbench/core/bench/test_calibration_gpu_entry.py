"""Verify managed and explicitly unlocked calibration with CPU substitutes."""

from types import SimpleNamespace

import pytest

from sol_execbench.core.bench import gpu_execution, host_clock_guard
from sol_execbench.core.bench.clock_ownership import ClockDevice
from sol_execbench.core.bench.gpu_lock import require_supervised_gpu_execution

UUID = "11111111-2222-3333-4444-555555555555"
SCRIPTS = "scripts/internal/"


@pytest.fixture
def hardware(monkeypatch, native_gpu_cpu_observation):
    device = ClockDevice(bdf="0000:05:00.0", uuid=UUID)
    writes = []
    monkeypatch.setattr(
        host_clock_guard, "observe_clock_devices", lambda: (device,)
    )
    monkeypatch.setattr(
        host_clock_guard,
        "observe_clock_levels",
        lambda *_: ("AMDSMI_DEV_PERF_LEVEL_AUTO",),
    )
    monkeypatch.setattr(
        host_clock_guard, "set_peak_clocks", lambda *_: writes.append("PEAK")
    )
    monkeypatch.setattr(
        host_clock_guard,
        "unlock_clocks",
        lambda *_: writes.append("AUTO") or True,
    )
    monkeypatch.setattr(
        gpu_execution, "synchronize_initialized_device", lambda: None
    )
    return writes


@pytest.mark.parametrize("entry", ["resource", "qualified"])
@pytest.mark.parametrize("allow_unlocked", [False, True])
def test_calibration_uses_requested_clock_ownership(
    entry, allow_unlocked, hardware, load_script, monkeypatch, tmp_path
):
    name = (
        "run_rdna4_resource_peak_calibration.py"
        if entry == "resource"
        else "run_qualified_rdna4_resource_peak_calibration.py"
    )
    module = load_script(SCRIPTS + "rdna4/" + name)
    args = SimpleNamespace(
        workdir=tmp_path, stage="run", allow_unlocked=allow_unlocked
    )
    monkeypatch.setattr(module, "_parse_args", lambda *_: args)

    def payload(*_):
        require_supervised_gpu_execution()
        assert hardware == ([] if allow_unlocked else ["PEAK"])
        return 0

    monkeypatch.setattr(
        module,
        "_calibrate" if entry == "resource" else "_run_calibration",
        payload,
    )
    assert module.main() == 0
    assert hardware == ([] if allow_unlocked else ["PEAK", "AUTO"])
