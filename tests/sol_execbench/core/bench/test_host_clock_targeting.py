# SPDX-License-Identifier: Apache-2.0
"""UUID host control must retain identity, ownership and teardown boundaries."""

import json
import subprocess
from typing import TypedDict

import pytest

from sol_execbench.core.bench import (
    clock_lock,
    clock_ownership,
    host_clock_guard,
)
from sol_execbench.core.platform.clock_commands import ENV_HOST_CLOCK_GPU_UUID

TARGET_UUID = "a3007590-0000-1000-800f-a29c1cca1511"
OTHER_UUID = "000013c0-0000-1000-8000-000000000000"
TARGET = clock_ownership.ClockDevice(bdf="0000:05:00.0", uuid=TARGET_UUID)
OTHER = clock_ownership.ClockDevice(bdf="0000:14:00.0", uuid=OTHER_UUID)


class TargetHardware(TypedDict):
    levels: dict[str, str]
    commands: list[list[str]]
    setup_failure: bool
    restore_failure: bool
    extra_readback: bool
    workloads: int


@pytest.fixture
def hardware(monkeypatch, tmp_path):
    monkeypatch.setenv("GPU_LOCK_UUID", TARGET_UUID)
    monkeypatch.setenv("GPU_LOCK_DIR", str(tmp_path))
    monkeypatch.setenv("SOL_EXECBENCH_CLOCK_STATE_DIR", str(tmp_path))
    monkeypatch.delenv(clock_ownership.ENV_CLOCK_OWNER, raising=False)
    monkeypatch.delenv(ENV_HOST_CLOCK_GPU_UUID, raising=False)
    monkeypatch.setattr(
        host_clock_guard, "observe_clock_devices", lambda: (TARGET, OTHER)
    )
    monkeypatch.setattr(
        clock_lock,
        "_amd_smi_executable",
        lambda _gpu_uuid=None: "/opt/rocm/bin/amd-smi",
    )
    monkeypatch.setattr(clock_lock.time, "sleep", lambda _: None)
    state: TargetHardware = {
        "levels": {TARGET_UUID: "AUTO", OTHER_UUID: "AUTO"},
        "commands": [],
        "setup_failure": False,
        "restore_failure": False,
        "extra_readback": False,
        "workloads": 0,
    }

    def run(command, **_kwargs):
        state["commands"].append(command)
        if "set" in command:
            assert command[:4] == ["sudo", "-n", "/opt/rocm/bin/amd-smi", "set"]
            assert command[4] == "-l"
            assert command[6:] == ["-g", TARGET_UUID]
            level = command[5]
            records = list((tmp_path / "clock-leases").glob("*.json"))
            assert len(records) == 1
            record = clock_ownership.ClockOwnerRecord.model_validate_json(
                records[0].read_text()
            )
            clock_ownership.verify_clock_owner(
                tmp_path, record.lease_id, (TARGET,)
            )
            assert record.devices == (TARGET,)
            if level == "AUTO" and state["restore_failure"]:
                raise subprocess.CalledProcessError(1, command)
            state["levels"][TARGET_UUID] = level
            if level == "STABLE_PEAK" and state["setup_failure"]:
                raise subprocess.CalledProcessError(1, command)
            return subprocess.CompletedProcess(command, 0, "", "")
        assert command == [
            "/opt/rocm/bin/amd-smi",
            "metric",
            "-l",
            "--json",
            "--gpu",
            TARGET_UUID,
        ]
        rows = [
            {
                "gpu": 0,
                "perf_level": f"AMDSMI_DEV_PERF_LEVEL_{state['levels'][TARGET_UUID]}",
            }
        ]
        if state["extra_readback"]:
            rows.append(rows[0].copy())
        return subprocess.CompletedProcess(
            command, 0, json.dumps({"gpu_data": rows}), ""
        )

    monkeypatch.setattr(clock_lock.subprocess, "run", run)
    return state


def owner_record(tmp_path):
    (path,) = (tmp_path / "clock-leases").glob("*.json")
    return clock_ownership.ClockOwnerRecord.model_validate_json(
        path.read_text()
    )


@pytest.mark.parametrize(
    "failure", [False, True], ids=["normal", "workload-error"]
)
def test_uuid_window_uses_exact_commands_and_restores_only_owned_gpu(
    hardware, tmp_path, failure
):
    def runner(command, environment):
        hardware["workloads"] += 1
        assert environment["SOL_EXECBENCH_CLOCKS_LOCKED"] == "1"
        assert environment[ENV_HOST_CLOCK_GPU_UUID] == TARGET_UUID
        assert hardware["levels"] == {
            TARGET_UUID: "STABLE_PEAK",
            OTHER_UUID: "AUTO",
        }
        record = clock_ownership.verify_clock_owner(
            tmp_path, environment[clock_ownership.ENV_CLOCK_OWNER], (TARGET,)
        )
        assert record.container_supervised
        if failure:
            raise RuntimeError("workload failed")
        return subprocess.CompletedProcess(command, 0)

    def invoke():
        return host_clock_guard.run_with_host_clock_guard(
            ["docker", "run"],
            gpu_uuid=f" {TARGET_UUID.upper()} ",
            runner=runner,
        )

    if failure:
        with pytest.raises(RuntimeError, match="workload failed"):
            invoke()
    else:
        assert invoke() == 0
    assert hardware["workloads"] == 1
    assert hardware["levels"] == {TARGET_UUID: "AUTO", OTHER_UUID: "AUTO"}
    assert (
        owner_record(tmp_path).state == clock_ownership.ClockOwnerState.RESTORED
    )


@pytest.mark.parametrize(
    "initial", ["STABLE_PEAK", "MANUAL"], ids=["external-peak", "manual"]
)
def test_target_requires_initial_auto(hardware, tmp_path, initial):
    hardware["levels"][TARGET_UUID] = initial
    with (
        pytest.raises(
            clock_ownership.ClockOwnershipError,
            match="initial_policy_unverified",
        ),
        host_clock_guard.host_clock_window(
            gpu_uuid=TARGET_UUID, container_supervised=True
        ),
    ):
        pytest.fail("must not start workload")
    assert not any("set" in cmd for cmd in hardware["commands"])
    assert not (tmp_path / "clock-leases").exists()


@pytest.mark.parametrize(
    "uuid",
    ["bad", "", "aaaaaaaa-0000-1000-8000-000000000000"],
    ids=["malformed", "empty", "unknown"],
)
def test_invalid_target_never_mutates(hardware, uuid):
    with (
        pytest.raises((ValueError, clock_ownership.ClockOwnershipError)),
        host_clock_guard.host_clock_window(
            gpu_uuid=uuid, container_supervised=True
        ),
    ):
        pytest.fail("must not start workload")
    assert hardware["commands"] == []


def test_ambiguous_target_never_mutates(hardware, monkeypatch):
    monkeypatch.setattr(
        host_clock_guard, "observe_clock_devices", lambda: (TARGET, TARGET)
    )
    with (
        pytest.raises(
            clock_ownership.ClockOwnershipError, match="identity_mismatch"
        ),
        host_clock_guard.host_clock_window(
            gpu_uuid=TARGET_UUID, container_supervised=True
        ),
    ):
        pytest.fail("must not start workload")
    assert hardware["commands"] == []


def test_target_readback_rejects_multiple_devices(hardware):
    hardware["extra_readback"] = True
    assert not clock_lock.verify_clocks(TARGET_UUID)
    with (
        pytest.raises(
            clock_ownership.ClockOwnershipError,
            match="initial_policy_unverified",
        ),
        host_clock_guard.host_clock_window(
            gpu_uuid=TARGET_UUID, container_supervised=True
        ),
    ):
        pytest.fail("must not start workload")
    assert not any("set" in cmd for cmd in hardware["commands"])


@pytest.mark.parametrize(
    "restore_failure",
    [False, True],
    ids=["rollback-verified", "rollback-failed"],
)
def test_target_setup_failure_never_runs_unlocked(
    hardware, tmp_path, restore_failure
):
    hardware["setup_failure"] = True
    hardware["restore_failure"] = restore_failure
    expected = (
        clock_lock.ClockRestoreError
        if restore_failure
        else clock_lock.ClockSetupError
    )
    with (
        pytest.raises(expected),
        host_clock_guard.host_clock_window(
            gpu_uuid=TARGET_UUID, container_supervised=True
        ),
    ):
        pytest.fail("must not start workload")
    record = owner_record(tmp_path)
    assert record.state == ("restore_failed" if restore_failure else "restored")
    assert not record.container_supervised
    assert hardware["levels"][OTHER_UUID] == "AUTO"


@pytest.mark.parametrize(
    "explicit", [False, True], ids=["environment", "override"]
)
def test_uuid_from_environment_reaches_production_window(hardware, explicit):
    assert (
        host_clock_guard.run_with_host_clock_guard(
            ["true"],
            environment={
                ENV_HOST_CLOCK_GPU_UUID: OTHER_UUID if explicit else TARGET_UUID
            },
            gpu_uuid=TARGET_UUID if explicit else None,
            runner=lambda command, _environment: subprocess.CompletedProcess(
                command, 0
            ),
        )
        == 0
    )
    assert [cmd[5] for cmd in hardware["commands"] if "set" in cmd] == [
        "STABLE_PEAK",
        "AUTO",
    ]


@pytest.mark.parametrize(
    "natural_clocks", [False, True], ids=["managed", "natural"]
)
def test_cli_forwards_uuid(monkeypatch, natural_clocks):
    # Exercise parser forwarding without launching an external process.
    observed = {}

    def invoke(command, *, gpu_uuid, natural_clocks):
        observed.update(
            command=command, gpu_uuid=gpu_uuid, natural_clocks=natural_clocks
        )
        return 0

    monkeypatch.setattr(host_clock_guard, "run_with_host_clock_guard", invoke)
    arguments = ["--gpu-uuid", TARGET_UUID]
    if natural_clocks:
        arguments.append("--natural-clocks")
    assert host_clock_guard.main([*arguments, "--", "true"]) == 0
    assert observed == {
        "command": ["true"],
        "gpu_uuid": TARGET_UUID,
        "natural_clocks": natural_clocks,
    }


@pytest.mark.parametrize(
    "readback_ok", [True, False], ids=["auto", "readback-failure"]
)
def test_natural_container_retains_supervision_in_final_receipt(
    hardware, tmp_path, readback_ok
):
    def runner(command, environment):
        assert environment["SOL_EXECBENCH_CLOCKS_LOCKED"] == "0"
        assert owner_record(tmp_path).container_supervised
        if not readback_ok:
            hardware["levels"][TARGET_UUID] = "MANUAL"
        return subprocess.CompletedProcess(command, 0)

    def invoke():
        return host_clock_guard.run_with_host_clock_guard(
            ["docker", "run"], runner=runner, natural_clocks=True
        )

    if readback_ok:
        assert invoke() == 0
    else:
        with pytest.raises(clock_lock.ClockRestoreError):
            invoke()
    record = owner_record(tmp_path)
    assert record.container_supervised
    expected = (
        clock_ownership.ClockOwnerState.RESTORED
        if readback_ok
        else clock_ownership.ClockOwnerState.RESTORE_FAILED
    )
    assert record.state == expected
    assert all("set" not in command for command in hardware["commands"])
