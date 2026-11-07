"""Persist command failures without changing the public clock lifecycle."""

import json
import logging
import subprocess

import pytest

from sol_execbench.core.bench import clock_lock
from sol_execbench.core.platform.clock_commands import ENV_HOST_AMD_SMI

UUID = "a3007590-0000-1000-800f-a29c1cca1511"


@pytest.fixture
def diagnostic_log(tmp_path):
    path = tmp_path / "native-stderr.log"
    handler = logging.FileHandler(path)
    root = logging.getLogger()
    root.addHandler(handler)
    try:
        yield path
    finally:
        root.removeHandler(handler)
        handler.close()


def diagnostics(path):
    return [
        json.loads(line.split("clock_command_diagnostic ", 1)[1])
        for line in path.read_text().splitlines()
        if "clock_command_diagnostic " in line
    ]


def test_setup_failure_persists_bounded_redacted_output(
    diagnostic_log, monkeypatch
):
    monkeypatch.setenv("API_TOKEN", "never-log-this-environment-secret")
    monkeypatch.setenv("ROCM_PATH", "/fixture/rocm")
    monkeypatch.setenv(ENV_HOST_AMD_SMI, "/fixture/amd-smi")

    def fail(command, **_):
        raise subprocess.CalledProcessError(
            1,
            command,
            output="setup stdout",
            stderr="x" * 5000 + " password=secret-value\nsetup rejected",
        )

    monkeypatch.setattr(clock_lock.subprocess, "run", fail)
    with pytest.raises(clock_lock.ClockSetupError):
        clock_lock.set_peak_clocks(UUID)
    (event,) = diagnostics(diagnostic_log)
    assert event["phase"] == "setup"
    assert event["returncode"] == 1
    assert event["stdout"] == "setup stdout"
    assert event["stderr"].endswith("password=<redacted>\nsetup rejected")
    assert len(event["stderr"]) <= 4096
    assert event["stderr_truncated"]
    assert event["command"] == [
        "sudo",
        "-n",
        "/fixture/amd-smi",
        "set",
        "-l",
        "STABLE_PEAK",
        "-g",
        UUID,
    ]
    assert event["environment"]["ROCM_PATH"] == "/fixture/rocm"
    text = diagnostic_log.read_text()
    assert "never-log-this-environment-secret" not in text
    assert "secret-value" not in text
    assert "NoNewPrivs" in event["process_status"]


def test_restore_failure_persists_output_and_still_returns_false(
    diagnostic_log, monkeypatch
):
    monkeypatch.setenv(ENV_HOST_AMD_SMI, "/fixture/amd-smi")

    def fail(command, **_):
        raise subprocess.CalledProcessError(
            1, command, output="restore stdout", stderr="restore rejected"
        )

    monkeypatch.setattr(clock_lock.subprocess, "run", fail)
    assert clock_lock.unlock_clocks(UUID) is False
    (event,) = diagnostics(diagnostic_log)
    assert event["phase"] == "restore"
    assert event["returncode"] == 1
    assert event["stdout"] == "restore stdout"
    assert event["stderr"] == "restore rejected"
    assert event["command"] == [
        "sudo",
        "-n",
        "/fixture/amd-smi",
        "set",
        "-l",
        "AUTO",
        "-g",
        UUID,
    ]


def test_successful_commands_are_persisted_without_changing_readback(
    diagnostic_log, monkeypatch
):
    monkeypatch.setenv(ENV_HOST_AMD_SMI, "/fixture/amd-smi")
    monkeypatch.setattr(clock_lock.time, "sleep", lambda _: None)
    levels = []

    def run(command, **_):
        if command[0] == "sudo":
            levels.append(command[5])
            return subprocess.CompletedProcess(
                command, 0, "accepted", "tool note"
            )
        return subprocess.CompletedProcess(
            command,
            0,
            json.dumps(
                {
                    "gpu_data": [
                        {
                            "gpu": 0,
                            "perf_level": "AMDSMI_DEV_PERF_LEVEL_" + levels[-1],
                        }
                    ]
                }
            ),
            "",
        )

    monkeypatch.setattr(clock_lock.subprocess, "run", run)
    assert clock_lock.set_peak_clocks(UUID) is None
    assert clock_lock.unlock_clocks(UUID) is True
    events = diagnostics(diagnostic_log)
    assert [x["phase"] for x in events] == ["setup", "restore"]
    assert levels == ["STABLE_PEAK", "AUTO"]
    assert all(
        x["returncode"] == 0
        and x["stdout"] == "accepted"
        and x["stderr"] == "tool note"
        for x in events
    )


def test_zero_exit_failure_marker_remains_failure_and_is_logged(
    diagnostic_log, monkeypatch
):
    monkeypatch.setenv(ENV_HOST_AMD_SMI, "/fixture/amd-smi")
    monkeypatch.setattr(
        clock_lock.subprocess,
        "run",
        lambda command, **_: subprocess.CompletedProcess(
            command,
            0,
            "ERROR: Unable to set performance level",
            "vendor diagnostic",
        ),
    )
    assert clock_lock.unlock_clocks(UUID) is False
    (event,) = diagnostics(diagnostic_log)
    assert event["returncode"] == 0
    assert event["error"] == "output_failure_marker"
    assert event["stderr"] == "vendor diagnostic"


@pytest.mark.parametrize(
    "failure", ["timeout", "missing"], ids=["timeout", "missing-executable"]
)
def test_setup_timeout_or_missing_executable_preserves_failure(
    diagnostic_log, monkeypatch, failure
):
    monkeypatch.setenv(ENV_HOST_AMD_SMI, "/fixture/amd-smi")

    def fail(command, **_):
        if failure == "timeout":
            raise subprocess.TimeoutExpired(
                command, 30, output=b"partial stdout", stderr=b"partial stderr"
            )
        raise FileNotFoundError("fixture executable missing")

    monkeypatch.setattr(clock_lock.subprocess, "run", fail)
    with pytest.raises(clock_lock.ClockSetupError):
        clock_lock.set_peak_clocks(UUID)
    (event,) = diagnostics(diagnostic_log)
    assert event["returncode"] is None
    assert event["error"] == (
        "TimeoutExpired" if failure == "timeout" else "FileNotFoundError"
    )
    if failure == "timeout":
        assert event["stdout"] == "partial stdout"
        assert event["stderr"] == "partial stderr"
