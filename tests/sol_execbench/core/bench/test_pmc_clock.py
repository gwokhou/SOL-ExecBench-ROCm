# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
"""Target-scoped PMC policy setup/teardown, without real privileged writes."""

import json
import os
import signal
import subprocess
from typing import TypedDict

import pytest

from sol_execbench.core.bench.clock_ownership import (
    LEASE_DIRECTORY,
    ClockOwnerRecord,
    ClockOwnership,
    ClockOwnerState,
)
from sol_execbench.core.bench.rocm_profiler import pmc_clock


class ControlState(TypedDict):
    level: str
    calls: list[list[str]]
    fail_peak: bool
    fail_restore: bool


@pytest.fixture
def control(monkeypatch, tmp_path):
    monkeypatch.setenv("SOL_EXECBENCH_GPU_LOCK_DIR", str(tmp_path / "locks"))
    monkeypatch.delenv("SOL_EXECBENCH_CLOCK_OWNER", raising=False)
    state: ControlState = {
        "level": "AUTO",
        "calls": [],
        "fail_peak": False,
        "fail_restore": False,
    }
    monkeypatch.setenv(
        pmc_clock.ENV_ROCR_VISIBLE_DEVICES, pmc_clock.TARGET_ROCR_UUID
    )
    monkeypatch.setattr(pmc_clock, "clock_executable", lambda: "amd-smi")

    def run(command, **_kwargs):
        state["calls"].append(command)
        code = 0
        if "list" in command:
            payload = [
                {
                    "gpu": 0,
                    "bdf": pmc_clock.TARGET_BDF,
                    "uuid": pmc_clock.TARGET_SMI_UUID,
                }
            ]
        elif "set" in command:
            level = command[command.index("-l") + 1]
            if level == "STABLE_PEAK":
                state["level"] = level
                code = int(state["fail_peak"])
            elif state["fail_restore"]:
                code = 1
            else:
                state["level"] = level
            payload = {}
        else:
            payload = {
                "gpu_data": [
                    {
                        "gpu": 0,
                        "perf_level": f"AMDSMI_DEV_PERF_LEVEL_{state['level']}",
                    }
                ]
            }
        return subprocess.CompletedProcess(
            command, code, json.dumps(payload), ""
        )

    monkeypatch.setattr(pmc_clock, "run_in_process_group_bounded", run)
    return state


def receipt(tmp_path):
    paths = list((tmp_path / "profile.pmc-clock").glob("*.json"))
    assert len(paths) == 1
    return pmc_clock.PMCClockReceipt.model_validate_json(paths[0].read_text())


@pytest.mark.parametrize(
    "failure", ["none", "exception", "timeout", "interrupt", "sigterm"], ids=str
)
def test_owned_window_restores_auto_on_every_exit(tmp_path, control, failure):
    previous = signal.getsignal(signal.SIGTERM)

    def collect():
        with pmc_clock.pmc_clock_window(tmp_path / "profile"):
            assert control["level"] == "STABLE_PEAK"
            if failure == "exception":
                raise ValueError("collection failed")
            if failure == "timeout":
                raise subprocess.TimeoutExpired("rocprofv3", 1)
            if failure == "interrupt":
                raise KeyboardInterrupt
            if failure == "sigterm":
                os.kill(os.getpid(), signal.SIGTERM)

    expected = {
        "exception": ValueError,
        "timeout": subprocess.TimeoutExpired,
        "interrupt": KeyboardInterrupt,
        "sigterm": SystemExit,
    }
    if failure == "none":
        collect()
    else:
        with pytest.raises(expected[failure]):
            collect()
    assert control["level"] == "AUTO"
    assert signal.getsignal(signal.SIGTERM) == previous
    observed = receipt(tmp_path)
    assert observed.auto_verified and observed.peak_verified
    assert observed.collection_completed == (failure == "none")
    for command in control["calls"]:
        if "metric" in command:
            assert command[command.index("--gpu") + 1] == pmc_clock.TARGET_BDF


def test_partial_setup_failure_still_restores(tmp_path, control):
    control["fail_peak"] = True
    with (
        pytest.raises(RuntimeError, match="pmc_clock_command_failed"),
        pmc_clock.pmc_clock_window(tmp_path / "profile"),
    ):
        pytest.fail("must not collect without verified STABLE_PEAK")
    assert receipt(tmp_path).auto_verified
    assert control["level"] == "AUTO"


def test_restore_failure_is_not_success(tmp_path, control):
    control["fail_restore"] = True
    with (
        pytest.raises(RuntimeError, match="failed to reset and verify"),
        pmc_clock.pmc_clock_window(tmp_path / "profile"),
    ):
        pass
    assert not receipt(tmp_path).auto_verified
    assert control["level"] == "STABLE_PEAK"


def test_external_policy_is_preserved(tmp_path, control):
    control["level"] = "STABLE_PEAK"
    with (
        pytest.raises(RuntimeError, match="pmc_clock_readback_failed"),
        pmc_clock.pmc_clock_window(tmp_path / "profile"),
    ):
        pytest.fail("must refuse an external policy")
    assert not any("set" in command for command in control["calls"])


def test_other_device_selection_has_no_effect(tmp_path, control, monkeypatch):
    monkeypatch.setenv(pmc_clock.ENV_ROCR_VISIBLE_DEVICES, "GPU-other")
    with pmc_clock.pmc_clock_window(tmp_path / "profile"):
        pass
    assert control["calls"] == []
    assert list(tmp_path.iterdir()) == []


def test_repeated_windows_keep_separate_receipts(tmp_path, control):
    for _ in range(2):
        with pmc_clock.pmc_clock_window(tmp_path / "profile"):
            pass
    assert len(list((tmp_path / "profile.pmc-clock").glob("*.json"))) == 2


def test_best_effort_requires_successful_rollback(tmp_path, control):
    control["fail_peak"] = True
    options = pmc_clock.PMCClockOptions(mode=pmc_clock.PMCClockMode.BEST_EFFORT)
    with pmc_clock.pmc_clock_window(tmp_path / "profile", options) as lease:
        assert not lease.locked
        assert control["level"] == "AUTO"
    assert receipt(tmp_path).auto_verified


def test_best_effort_never_hides_failed_restore(tmp_path, control):
    control.update(fail_peak=True, fail_restore=True)
    options = pmc_clock.PMCClockOptions(mode=pmc_clock.PMCClockMode.BEST_EFFORT)
    with (
        pytest.raises(RuntimeError, match="failed to reset and verify"),
        pmc_clock.pmc_clock_window(tmp_path / "profile", options),
    ):
        pytest.fail("collection must not continue after failed restore")


def test_all_devices_is_explicit(tmp_path, control):
    with pmc_clock.pmc_clock_window(
        tmp_path / "profile", pmc_clock.PMCClockOptions(target="all")
    ):
        pass
    mutations = [cmd for cmd in control["calls"] if "set" in cmd]
    assert len(mutations) == 2
    assert all(
        cmd[cmd.index("-g") + 1] == pmc_clock.TARGET_SMI_UUID
        for cmd in mutations
    )


def test_environment_options_are_normalized(monkeypatch):
    monkeypatch.setenv(pmc_clock.ENV_PMC_CLOCK_TARGET, "all")
    monkeypatch.setenv(pmc_clock.ENV_PMC_CLOCK_MODE, "best-effort")
    assert (
        pmc_clock.configured_pmc_clock_options()
        == pmc_clock.PMCClockOptions(
            target="all", mode=pmc_clock.PMCClockMode.BEST_EFFORT
        )
    )
    monkeypatch.setenv(pmc_clock.ENV_PMC_CLOCK_MODE, "ignore-restoration")
    with pytest.raises(ValueError, match="not a valid PMCClockMode"):
        pmc_clock.configured_pmc_clock_options()


def test_archive_requires_current_schema(tmp_path, control):
    with pmc_clock.pmc_clock_window(tmp_path / "profile"):
        pass
    payload = receipt(tmp_path).model_dump(mode="json")
    payload["schema_version"] = "obsolete"
    with pytest.raises(ValueError, match="requires schema_version"):
        pmc_clock.PMCClockReceipt.model_validate(payload)


def test_unavailable_profiler_does_not_change_policy(
    tmp_path, control, monkeypatch
):
    from sol_execbench.cli.evaluation import command

    monkeypatch.setattr(command, "_replay_snapshot", lambda _: None)
    _, result = command._run_profiled_evaluation(
        ["python", "candidate.py"],
        staging_dir=tmp_path,
        output_file=tmp_path / "trace.jsonl",
        timeout=5,
        rocprofv3_available=False,
        counter_mode=True,
    )
    assert not result.succeeded
    assert control["calls"] == []


def test_setup_readback_failure_restores_auto(tmp_path, control, monkeypatch):
    original = pmc_clock.PMCClockControl.verify

    def verify(self, level):
        if level == pmc_clock.ClockLevel.PEAK:
            raise RuntimeError("peak readback unavailable")
        original(self, level)

    monkeypatch.setattr(pmc_clock.PMCClockControl, "verify", verify)
    with (
        pytest.raises(RuntimeError, match="peak readback unavailable"),
        pmc_clock.pmc_clock_window(tmp_path / "profile"),
    ):
        pytest.fail("no verified peak")
    assert receipt(tmp_path).auto_verified


@pytest.mark.parametrize("fails", [False, True], ids=["success", "inner-error"])
def test_nested_pmc_borrows_without_resetting_owner(tmp_path, control, fails):
    def inner():
        with pmc_clock.pmc_clock_window(tmp_path / "inner") as borrowed:
            assert borrowed.locked and not borrowed.acquired
            if fails:
                raise ValueError("inner failed")

    with pmc_clock.pmc_clock_window(tmp_path / "profile"):
        if fails:
            with pytest.raises(ValueError, match="inner failed"):
                inner()
        else:
            inner()
        assert control["level"] == "STABLE_PEAK"
        assert len([cmd for cmd in control["calls"] if "set" in cmd]) == 1
        borrowed_path = next((tmp_path / "inner.pmc-clock").glob("*.json"))
        borrowed = pmc_clock.PMCClockReceipt.model_validate_json(
            borrowed_path.read_text()
        )
        assert borrowed.ownership == ClockOwnership.BORROWED
        assert borrowed.borrow_ended and not borrowed.auto_verified
    assert control["level"] == "AUTO"
    owner_path = (
        tmp_path / "locks" / LEASE_DIRECTORY / f"{borrowed.owner_lease_id}.json"
    )
    assert (
        ClockOwnerRecord.model_validate_json(owner_path.read_text()).state
        == ClockOwnerState.RESTORED
    )


def test_best_effort_rejects_unverifiable_owner(tmp_path, control, monkeypatch):
    monkeypatch.setenv(pmc_clock.ENV_CLOCK_OWNER, "a" * 32)
    control["level"] = "STABLE_PEAK"
    with (
        pytest.raises(pmc_clock.ClockOwnershipError, match="unavailable"),
        pmc_clock.pmc_clock_window(
            tmp_path / "profile",
            pmc_clock.PMCClockOptions(mode=pmc_clock.PMCClockMode.BEST_EFFORT),
        ),
    ):
        pytest.fail("an environment token is not ownership")
    assert not any("set" in cmd for cmd in control["calls"])


def test_host_guard_owns_pmc_borrow_and_final_auto(
    tmp_path, control, monkeypatch
):
    from sol_execbench.core.bench import clock_ownership, host_clock_guard

    restores = []

    def restore():
        restores.append("host")
        control["level"] = "AUTO"
        return True

    def activate():
        control["level"] = "STABLE_PEAK"

    monkeypatch.setattr(host_clock_guard, "set_peak_clocks", activate)
    monkeypatch.setattr(host_clock_guard, "unlock_clocks", restore)
    monkeypatch.setattr(
        host_clock_guard,
        "observe_clock_levels",
        lambda: (f"AMDSMI_DEV_PERF_LEVEL_{control['level']}",),
    )
    monkeypatch.setattr(
        host_clock_guard,
        "observe_clock_devices",
        lambda: (
            clock_ownership.ClockDevice(
                bdf=pmc_clock.TARGET_BDF, uuid=pmc_clock.TARGET_SMI_UUID
            ),
        ),
    )

    def runner(command, environment):
        assert environment[pmc_clock.ENV_CLOCK_OWNER]
        with pmc_clock.pmc_clock_window(tmp_path / "profile") as borrowed:
            assert borrowed.locked and not borrowed.acquired
        assert restores == [] and control["level"] == "STABLE_PEAK"
        return subprocess.CompletedProcess(command, 0)

    assert (
        host_clock_guard.run_with_host_clock_guard(
            ("fake-container",), runner=runner
        )
        == 0
    )
    assert restores == ["host"] and control["level"] == "AUTO"
    observed = receipt(tmp_path)
    assert observed.ownership == ClockOwnership.BORROWED
    assert observed.borrow_ended and not observed.auto_verified


@pytest.mark.parametrize(
    "fails", [False, True], ids=["success", "collector-error"]
)
def test_cli_replay_restores_policy_and_canonical_timing(
    tmp_path, control, monkeypatch, fails
):
    from sol_execbench.cli.evaluation import command, runtime
    from sol_execbench.core.bench.performance_model.timing_evidence import (
        RAW_TIMING_FILENAME,
    )
    from sol_execbench.core.bench.rocm_profiler.models import (
        Rocprofv3ProfileResult,
        Rocprofv3ProfileStatus,
    )

    timing = tmp_path / RAW_TIMING_FILENAME
    timing.write_bytes(b"canonical samples")

    def collect(request, **_kwargs):
        assert control["level"] == "STABLE_PEAK"
        timing.write_bytes(b"replay must never become canonical")
        if fails:
            raise RuntimeError("collector failure")
        return Rocprofv3ProfileResult(
            status=Rocprofv3ProfileStatus.SUCCESS,
            command=("rocprofv3",),
            output_directory=request.output_directory,
            output_file=request.output_file,
        )

    class Packager:
        @staticmethod
        def restage_trusted_reference():
            pass

        @staticmethod
        def convert_stdout_to_traces(stdout: str):
            assert isinstance(stdout, str)
            return []

    monkeypatch.setattr(command, "collect_rocprofv3_counters", collect)
    monkeypatch.setattr(
        command, "resolve_rocm_tool", lambda _: tmp_path / "rocprofv3"
    )
    monkeypatch.setattr(command, "_replay_snapshot", lambda _: None)

    def replay():
        runtime._collect_counter_replay(
            Packager(),
            ["python", "candidate.py"],
            staging_dir=tmp_path,
            output_file=tmp_path / "trace.jsonl",
            timeout=5,
        )

    if fails:
        with pytest.raises(RuntimeError, match="collector failure"):
            replay()
    else:
        replay()
    assert control["level"] == "AUTO"
    assert timing.read_bytes() == b"canonical samples"


@pytest.mark.parametrize(
    "target", [pmc_clock.TARGET_BDF, pmc_clock.TARGET_SMI_UUID, "0", "all"]
)
def test_mutations_use_verified_uuid_and_authorized_arguments(
    tmp_path, control, target
):
    with pmc_clock.pmc_clock_window(
        tmp_path / "profile", pmc_clock.PMCClockOptions(target=target)
    ):
        pass
    assert [cmd for cmd in control["calls"] if "set" in cmd] == [
        [
            "sudo",
            "-n",
            "amd-smi",
            "set",
            "-l",
            level,
            "-g",
            pmc_clock.TARGET_SMI_UUID,
        ]
        for level in ("STABLE_PEAK", "AUTO")
    ]


def test_mutation_requires_verified_identity(tmp_path, control):
    controller = pmc_clock.PMCClockControl(
        tmp_path / "control", pmc_clock.PMCClockOptions()
    )
    with pytest.raises(
        pmc_clock.ClockOwnershipError, match="identity_unverified"
    ):
        controller.set_level(pmc_clock.ClockLevel.PEAK)
    assert not control["calls"]
