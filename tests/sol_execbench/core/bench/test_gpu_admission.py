"""CPU-only native host lifecycle against the neutral admission contract."""

import json
import subprocess
from pathlib import Path

import pytest

from sol_execbench.core.bench import (
    clock_ownership,
    clock_recovery,
    gpu_admission,
    host_clock_guard,
)
from sol_execbench.core.bench.gpu_admission import (
    Admission,
    AdmissionDevice,
    AdmissionError,
    AdmissionRecord,
)
from sol_execbench.core.bench.gpu_lock import acquire_gpu_lock
from sol_execbench.core.control_plane_schema_versions import (
    ExecutionControlSchema,
)
from sol_execbench.tools.docker_process import ContainerCleanupError

UUID = "a3007590-0000-1000-800f-a29c1cca1511"
DEVICE = {"pci_bdf": "0000:05:00.0", "smi_uuid": UUID}


def test_host_guard_finishes_native_release_before_ready(tmp_path, monkeypatch):
    admission = tmp_path / "admission.json"
    migration = tmp_path / "migration-check.json"
    migration.write_text('{"cpu_fixture":true,"explicit_clean_check":true}')
    admission.write_text(
        json.dumps(
            {
                "schema_version": "gpu-admission.v1",
                "device": DEVICE,
                "state": "ready",
                "owner": None,
                "record_ref": str(migration),
                "reason": None,
            }
        )
    )
    monkeypatch.setenv("GPU_LOCK_UUID", UUID)
    monkeypatch.setenv("GPU_LOCK_DIR", str(tmp_path / "lock"))
    monkeypatch.setenv(
        "SOL_EXECBENCH_CLOCK_STATE_DIR", str(tmp_path / "sol-private")
    )
    monkeypatch.setenv("SOL_EXECBENCH_ADMISSION_RECORD", str(admission))
    monkeypatch.setattr(
        host_clock_guard,
        "observe_clock_devices",
        lambda: (
            clock_ownership.ClockDevice(bdf=DEVICE["pci_bdf"], uuid=UUID),
        ),
    )
    monkeypatch.setattr(
        host_clock_guard,
        "observe_clock_levels",
        lambda *_: ("AMDSMI_DEV_PERF_LEVEL_AUTO",),
    )

    def active():
        record = AdmissionRecord.model_validate_json(admission.read_text())
        assert record.state == "held"
        assert record.record_ref is not None
        assert record.owner is not None
        native = json.loads(record.record_ref.read_text())
        assert native["state"] == "active"
        assert native["lease_id"] == record.owner.lease_id

    monkeypatch.setattr(
        host_clock_guard, "set_peak_clocks", lambda *_: active()
    )
    monkeypatch.setattr(
        host_clock_guard, "unlock_clocks", lambda *_: active() or True
    )

    def runner(command, environment):
        active()
        return subprocess.CompletedProcess(command, 23)

    assert (
        host_clock_guard.run_with_host_clock_guard(
            ("cpu-stub",), runner=runner, gpu_uuid=UUID
        )
        == 23
    )
    released = AdmissionRecord.model_validate_json(admission.read_text())
    assert released.state == "ready"
    assert released.owner is None
    assert released.record_ref is not None
    assert json.loads(released.record_ref.read_text())["state"] == "restored"


@pytest.fixture
def lifecycle(tmp_path, monkeypatch):
    admission = tmp_path / "admission.json"
    migration = tmp_path / "explicit-migration-check.json"
    migration.write_text(
        '{"cpu_fixture":true,"old_logs_checked":true,"clean":true}'
    )
    admission.write_text(
        json.dumps(
            {
                "schema_version": "gpu-admission.v1",
                "device": DEVICE,
                "state": "ready",
                "owner": None,
                "record_ref": str(migration),
                "reason": None,
            }
        )
    )
    monkeypatch.setenv("GPU_LOCK_UUID", UUID)
    monkeypatch.setenv("GPU_LOCK_DIR", str(tmp_path / "lock"))
    monkeypatch.setenv(
        "SOL_EXECBENCH_CLOCK_STATE_DIR", str(tmp_path / "private")
    )
    monkeypatch.setenv("SOL_EXECBENCH_ADMISSION_RECORD", str(admission))
    monkeypatch.delenv("GPU_LOCK_FD", raising=False)
    monkeypatch.delenv("SOL_EXECBENCH_CLOCK_OWNER", raising=False)
    device = clock_ownership.ClockDevice(bdf=DEVICE["pci_bdf"], uuid=UUID)
    monkeypatch.setattr(
        host_clock_guard, "observe_clock_devices", lambda: (device,)
    )
    monkeypatch.setattr(
        clock_recovery, "observe_clock_devices", lambda: (device,)
    )
    monkeypatch.setattr(
        host_clock_guard,
        "observe_clock_levels",
        lambda *_: ("AMDSMI_DEV_PERF_LEVEL_AUTO",),
    )
    effects = []

    def activate(*_):
        held = Admission(path=admission).read(AdmissionDevice(**DEVICE))
        assert held.state == "held"
        assert held.record_ref is not None
        assert json.loads(held.record_ref.read_text())["state"] == "active"
        effects.append("activate")

    monkeypatch.setattr(host_clock_guard, "set_peak_clocks", activate)
    monkeypatch.setattr(
        host_clock_guard,
        "unlock_clocks",
        lambda *_: effects.append("restore") or True,
    )

    def runner(command, environment):
        effects.append("work-and-cleanup")
        token = environment["SOL_EXECBENCH_CLOCK_OWNER"]
        path = clock_ownership.clock_owner_path(tmp_path / "lock", token)
        path.with_suffix(".container-id").write_text("a" * 64)
        return subprocess.CompletedProcess(command, 0)

    def run():
        return host_clock_guard.run_with_host_clock_guard(
            ("docker", "run", "cpu-fixture"), runner=runner, gpu_uuid=UUID
        )

    return admission, effects, run


@pytest.mark.parametrize(
    "case",
    [
        "missing",
        "unknown-schema",
        "wrong-device",
        "held",
        "blocked",
        "partial",
        "duplicate",
    ],
)
def test_unready_never_starts_device(lifecycle, case):
    path, effects, run = lifecycle
    value = json.loads(path.read_text())
    if case == "missing":
        path.unlink()
    elif case == "partial":
        path.write_text('{"state":')
    elif case == "duplicate":
        path.write_text(
            path.read_text().replace(
                '"state": "ready"', '"state": "held", "state": "ready"'
            )
        )
    else:
        if case == "unknown-schema":
            family, _, version = (
                ExecutionControlSchema.GPU_ADMISSION.rpartition(".v")
            )
            value["schema_version"] = f"{family}.v{int(version) + 1}"
        elif case == "wrong-device":
            value["device"]["pci_bdf"] = "0000:06:00.0"
        else:
            value.update(
                state=case,
                owner={"project": "kernel-cause", "lease_id": "prior"},
            )
            if case == "blocked":
                value["reason"] = "unknown_cleanup"
        path.write_text(json.dumps(value))
    with pytest.raises(AdmissionError):
        run()
    assert effects == []


def test_preflight_observes_without_reserving_and_does_not_read_foreign_log(
    lifecycle, tmp_path
):
    path, effects, _ = lifecycle
    original = path.read_bytes()
    lock = tmp_path / "lock"
    (lock / "clock-leases").mkdir(parents=True)
    (lock / "clock-leases/foreign.json").write_text("not our schema")
    with acquire_gpu_lock(
        timeout_seconds=0.1, admission_device=AdmissionDevice(**DEVICE)
    ):
        assert path.read_bytes() == original
    assert effects == []


def test_native_write_failure_keeps_held_without_activation(
    lifecycle, monkeypatch
):
    path, effects, run = lifecycle

    def fail(*_):
        raise OSError("native responsibility persistence failed")

    monkeypatch.setattr(clock_ownership, "write_clock_record", fail)
    with pytest.raises(OSError, match="native responsibility"):
        run()
    assert effects == []
    assert json.loads(path.read_text())["state"] == "held"


def test_private_directory_persistence_failure_stops_before_activation(
    lifecycle, monkeypatch
):
    path, effects, run = lifecycle

    def fail(_):
        raise OSError("private directory fsync failed")

    monkeypatch.setattr(clock_ownership, "ensure_durable_directory", fail)
    with pytest.raises(OSError, match="private directory fsync"):
        run()
    assert effects == []
    assert json.loads(path.read_text())["state"] == "held"


def test_native_final_failure_does_not_publish_ready(lifecycle, monkeypatch):
    path, effects, run = lifecycle
    writer = clock_ownership.write_clock_record

    def fail_final(target, record):
        if record.state == "restored":
            raise OSError("terminal persistence failed")
        writer(target, record)

    monkeypatch.setattr(clock_ownership, "write_clock_record", fail_final)
    with pytest.raises(OSError, match="terminal persistence"):
        run()
    assert effects == ["activate", "work-and-cleanup", "restore"]
    assert json.loads(path.read_text())["state"] == "held"


def test_container_cleanup_failure_prevents_restore_and_ready(
    lifecycle, monkeypatch
):
    path, effects, _ = lifecycle

    def runner(command, environment):
        raise ContainerCleanupError("still alive", "a" * 64)

    with pytest.raises(ContainerCleanupError):
        host_clock_guard.run_with_host_clock_guard(
            ("docker", "run"), runner=runner, gpu_uuid=UUID
        )
    assert effects == ["activate"]
    held = json.loads(path.read_text())
    assert held["state"] == "held"
    assert (
        json.loads(Path(held["record_ref"]).read_text())["state"]
        == "restore_failed"
    )


def test_restore_failure_remains_held_and_blocks_next_owner(
    lifecycle, monkeypatch
):
    path, effects, run = lifecycle
    monkeypatch.setattr(host_clock_guard, "unlock_clocks", lambda *_: False)
    with pytest.raises(RuntimeError):
        run()
    count = len(effects)
    with pytest.raises(AdmissionError):
        run()
    assert len(effects) == count
    assert json.loads(path.read_text())["state"] == "held"


@pytest.mark.parametrize("state", ["held", "ready"])
def test_neutral_publication_failure_is_not_success(
    lifecycle, monkeypatch, state
):
    path, effects, run = lifecycle
    writer = gpu_admission.durable_json_write

    def fail(target, value):
        if target == path and value["state"] == state:
            raise OSError("publication failed")
        writer(target, value)

    monkeypatch.setattr(gpu_admission, "durable_json_write", fail)
    with pytest.raises(OSError, match="publication failed"):
        run()
    observed = json.loads(path.read_text())["state"]
    if state == "held":
        assert effects == []
        assert observed == "ready"
    else:
        assert effects == ["activate", "work-and-cleanup", "restore"]
        assert observed == "blocked"


def test_only_original_sol_owner_can_confirm_recovery(
    lifecycle, monkeypatch, tmp_path
):
    path, _, run = lifecycle
    monkeypatch.setattr(host_clock_guard, "unlock_clocks", lambda *_: False)
    with pytest.raises(RuntimeError):
        run()
    held = json.loads(path.read_text())
    token = held["owner"]["lease_id"]
    calls = []
    monkeypatch.setattr(
        clock_recovery,
        "confirm_container_absent",
        lambda *_: calls.append("absent"),
    )

    def readback(command, *_):
        return (
            '[{"gpu":0,"process_list":[]}]'
            if command == "process"
            else '{"gpu_data":[{"gpu":0,"perf_level":"AMDSMI_DEV_PERF_LEVEL_AUTO"}]}'
        )

    monkeypatch.setattr(clock_recovery, "_read_device", readback)
    foreign = dict(held, owner={"project": "kernel-cause", "lease_id": token})
    path.write_text(json.dumps(foreign))
    with pytest.raises(AdmissionError, match="foreign"):
        clock_recovery.confirm_clock_recovery(tmp_path / "lock", token)
    assert calls == []
    path.write_text(json.dumps(held))
    clock_recovery.confirm_clock_recovery(tmp_path / "lock", token)
    assert json.loads(path.read_text())["state"] == "ready"
    native = Path(held["record_ref"])
    assert (native.parent / "recovery-history" / native.name).is_file()


def test_legacy_unknown_private_log_is_not_a_clean_migration(
    lifecycle, tmp_path
):
    _, effects, run = lifecycle
    native = tmp_path / "private/clock-leases"
    native.mkdir(parents=True)
    (native / "unknown.json").write_text('{"state":"restored"}')
    with pytest.raises(clock_ownership.ClockOwnershipError):
        run()
    assert effects == []


def test_explicit_uuid_required_in_neutral_mode(lifecycle, monkeypatch):
    _, effects, _ = lifecycle
    monkeypatch.delenv("GPU_LOCK_UUID")

    def unexpected_runner(_command, _environment):
        pytest.fail("missing UUID must stop before the workload")

    with pytest.raises(ValueError, match="GPU_LOCK_UUID"):
        host_clock_guard.run_with_host_clock_guard(
            ("cpu-fixture",), environment={}, runner=unexpected_runner
        )
    assert effects == []


def test_frozen_cross_project_structure_cases(tmp_path):
    cases = json.loads(
        (
            Path(__file__).resolve().parent
            / "fixtures/gpu_admission_schema_cases.json"
        ).read_text()
    )
    for case in cases:
        path = tmp_path / "neutral.json"
        path.write_text(json.dumps(case["record"]))
        if case["valid"]:
            assert (
                Admission(path=path).read(AdmissionDevice(**DEVICE)).state
                == case["record"]["state"]
            ), case["name"]
        else:
            with pytest.raises(AdmissionError):
                Admission(path=path).read(AdmissionDevice(**DEVICE))


@pytest.mark.parametrize("point", ["before-native", "before-ready"])
def test_interrupted_publication_leaves_held(lifecycle, monkeypatch, point):
    path, effects, run = lifecycle
    writer = gpu_admission.durable_json_write

    def interrupted(target, value):
        if (
            point == "before-native"
            and target != path
            and value.get("state") == "active"
        ) or (
            point == "before-ready"
            and target == path
            and value["state"] == "ready"
        ):
            raise SystemExit("CPU crash injection")
        writer(target, value)

    monkeypatch.setattr(gpu_admission, "durable_json_write", interrupted)
    monkeypatch.setattr(clock_ownership, "durable_json_write", interrupted)
    with pytest.raises(SystemExit, match="CPU crash"):
        run()
    held = json.loads(path.read_text())
    assert held["state"] == "held"
    if point == "before-native":
        assert effects == []
    else:
        assert (
            json.loads(Path(held["record_ref"]).read_text())["state"]
            == "restored"
        )


def test_direct_entry_cannot_pass_the_same_physical_lock(lifecycle):
    path, effects, run = lifecycle
    with (
        acquire_gpu_lock(
            timeout_seconds=0.1, admission_device=AdmissionDevice(**DEVICE)
        ),
        pytest.MonkeyPatch.context() as patch,
    ):
        patch.setenv("SOL_EXECBENCH_GPU_LOCK_TIMEOUT_SECONDS", "0.01")
        assert run() == 75
    assert effects == []
    assert json.loads(path.read_text())["state"] == "ready"


def test_identity_is_rechecked_under_lock(lifecycle, monkeypatch):
    _, effects, run = lifecycle
    observations = iter(
        [
            (clock_ownership.ClockDevice(bdf=DEVICE["pci_bdf"], uuid=UUID),),
            (clock_ownership.ClockDevice(bdf="0000:06:00.0", uuid=UUID),),
        ]
    )
    monkeypatch.setattr(
        host_clock_guard, "observe_clock_devices", lambda: next(observations)
    )
    with pytest.raises(AdmissionError, match="device_mismatch"):
        run()
    assert effects == []


def test_export_cpu_handoff(lifecycle, tmp_path):
    path, effects, run = lifecycle
    assert run() == 0
    released = json.loads(path.read_text())
    assert released["state"] == "ready"
    assert effects == ["activate", "work-and-cleanup", "restore"]
    (tmp_path / "interop.json").write_text(
        json.dumps(
            {
                "domain": "isolated-cpu-stub",
                "hardware_verified": False,
                "production_dispatch_allowed": False,
                "admission_record": str(path),
                "native_terminal_record": released["record_ref"],
                "physical_lock": str(
                    tmp_path
                    / "lock/gpu-11111111-2222-3333-4444-555555555555.L1-ownership.lock"
                ),
                "private_directory": str(tmp_path / "private"),
                "frozen_contract_identity": "2443d176d009b1b9241253687958bf97d4eee5d9479dcc01e07f846754e5437b",
                "next_project": "kernel-cause candidate must check and acquire the same lock",
            },
            indent=2,
        )
        + "\n"
    )
