"""Exercise public calibration boundaries with CPU hardware/payload substitutes."""

from functools import partial
from types import SimpleNamespace

import pytest

from sol_execbench.core.bench import gpu_execution, host_clock_guard
from sol_execbench.core.bench.clock_ownership import ClockDevice
from sol_execbench.core.bench.gpu_lock import (
    acquire_gpu_lock,
    require_supervised_gpu_execution,
)

UUID = "11111111-2222-3333-4444-555555555555"
SCRIPTS = "scripts/internal/"


class PayloadReached(Exception):
    """Stop before any compiler, profiler or compute operation."""


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


def _stub_calibration_preparation(module, entry, monkeypatch, payload):
    if entry == "resource":
        monkeypatch.setattr(module, "_prepare_calibration", lambda *_: object())
        monkeypatch.setattr(module, "_calibrate", payload)
    else:
        monkeypatch.setattr(
            module, "_verify_qualification", lambda *_, **__: None
        )
        monkeypatch.setattr(module, "_check_device", lambda *_: None)
        monkeypatch.setattr(
            module, "_prepare_qualified_calibration", lambda *_: object()
        )
        monkeypatch.setattr(
            module,
            "_legacy",
            lambda: {
                "_prepare_calibration": lambda *_: object(),
                "_calibrate": payload,
            },
        )


@pytest.mark.parametrize(
    "entry",
    [
        "resource",
        "qualified",
        "cdna",
        "diagnostic",
        "overhead",
        "aka",
        "clock-workload",
    ],
)
def test_calibration_payload_requires_native_l1_l2(
    entry, hardware, load_script, monkeypatch, tmp_path
):
    def payload(*_args, **_kwargs):
        require_supervised_gpu_execution()
        with (
            pytest.raises(TimeoutError),
            acquire_gpu_lock(timeout_seconds=0.01),
        ):
            pytest.fail("calibration released ownership before its payload")
        raise PayloadReached

    if entry in {"resource", "qualified"}:
        name = (
            "run_rdna4_resource_peak_calibration.py"
            if entry == "resource"
            else "run_qualified_rdna4_resource_peak_calibration.py"
        )
        module = load_script(SCRIPTS + "rdna4/" + name)
        monkeypatch.setattr(
            module,
            "_parse_args",
            lambda *_: SimpleNamespace(
                workdir=tmp_path,
                stage="run",
                allow_unlocked=False,
                check_only=False,
            ),
        )
        _stub_calibration_preparation(module, entry, monkeypatch, payload)
        invoke = module.main
    elif entry in {"cdna", "diagnostic"}:
        directory = "cdna3" if entry == "cdna" else "rdna4"
        module = load_script(
            f"{SCRIPTS}{directory}/run_{directory}_diagnostic_calibration.py"
        )
        monkeypatch.setattr(
            module, "resolve_rocm_tool", lambda _: tmp_path / "cpu-stub"
        )
        monkeypatch.setattr(module, "detect_rocm_device", payload)
        invoke = partial(
            module.run_calibration,
            output=tmp_path,
            **(
                {}
                if entry == "cdna"
                else {
                    "gpu_id": UUID,
                    "tuning_batches": 1,
                    "estimation_batches": 1,
                    "vram_policy_path": tmp_path,
                }
            ),
        )
    elif entry == "overhead":
        module = load_script(
            SCRIPTS + "rdna4/run_rdna4_profiler_overhead_calibration.py"
        )
        monkeypatch.setattr(module, "_prepare_environment", lambda _: object())
        monkeypatch.setattr(module, "_setup_calibration_clocks", payload)
        invoke = partial(
            module.run_calibration,
            module.CalibrationRequest(output_path=tmp_path),
        )
    elif entry == "aka":
        module = load_script(SCRIPTS + "aka_calibrate_tolerances.py")
        monkeypatch.setattr(
            module,
            "_parse_args",
            lambda: SimpleNamespace(
                stage="run", device="cuda:0", seed_count=2, repeats_per_seed=2
            ),
        )
        monkeypatch.setattr(module, "_require_qualification", payload)
        invoke = module.main
    else:
        module = load_script(
            SCRIPTS + "rdna4/rdna4_clock_lock_workload_test.py"
        )
        monkeypatch.setattr(module, "amd_smi", lambda *_: "CPU fixture")
        monkeypatch.setattr(module, "log", lambda *_: None)
        monkeypatch.setattr(module, "run_gpu_workload", payload)
        invoke = module.main
    with pytest.raises(PayloadReached):
        invoke()
    assert hardware == (
        ["PEAK", "AUTO"]
        if entry
        in {"resource", "qualified", "diagnostic", "overhead", "clock-workload"}
        else []
    )
    with acquire_gpu_lock(timeout_seconds=0.01):
        pass


@pytest.mark.parametrize(
    "script",
    [
        "rdna4/run_qualified_rdna4_resource_peak_calibration.py",
        "rdna4/run_rdna4_diagnostic_calibration.py",
        "aka_calibrate_tolerances.py",
    ],
)
def test_static_qualification_does_not_require_gpu(
    script, load_script, monkeypatch
):
    module = load_script(SCRIPTS + script)
    monkeypatch.delenv("GPU_LOCK_DIR", raising=False)
    monkeypatch.delenv("GPU_LOCK_UUID", raising=False)
    monkeypatch.setattr(
        gpu_execution,
        "resolve_gpu_device",
        lambda: pytest.fail("static qualification queried GPU"),
    )
    monkeypatch.setattr(
        module,
        "_parse_args",
        lambda *_: SimpleNamespace(
            stage="qualify-static",
            device="cuda:0",
            seed_count=2,
            repeats_per_seed=2,
            check_only=False,
        ),
    )

    def static(*_):
        raise PayloadReached

    monkeypatch.setattr(module, "_run_qualification", static)
    with pytest.raises(PayloadReached):
        module.main()


def test_explicit_calibration_uuid_conflict_stops_before_setter(
    hardware, load_script, tmp_path
):
    module = load_script(SCRIPTS + "rdna4/run_rdna4_diagnostic_calibration.py")
    with pytest.raises(ValueError, match="conflicts with GPU_LOCK_UUID"):
        module.run_calibration(
            output=tmp_path,
            gpu_id="99999999-2222-3333-4444-555555555555",
            tuning_batches=1,
            estimation_batches=1,
            vram_policy_path=tmp_path,
        )
    assert hardware == []


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
        workdir=tmp_path,
        stage="run",
        allow_unlocked=allow_unlocked,
        check_only=False,
    )
    monkeypatch.setattr(module, "_parse_args", lambda *_: args)

    def payload(*_):
        require_supervised_gpu_execution()
        assert hardware == ([] if allow_unlocked else ["PEAK"])
        return 0

    _stub_calibration_preparation(module, entry, monkeypatch, payload)
    assert module.main() == 0
    assert hardware == ([] if allow_unlocked else ["PEAK", "AUTO"])
