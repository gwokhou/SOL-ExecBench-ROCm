from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from sol_execbench.core.platform.clock_commands import ENV_HOST_AMD_SMI

SCRIPT_PATH = (
    Path(__file__).resolve().parents[4] / "scripts/setup_rocm_clock_sudoers.py"
)
SPEC = importlib.util.spec_from_file_location(
    "setup_rocm_clock_sudoers",
    SCRIPT_PATH,
)
assert SPEC is not None
assert SPEC.loader is not None
sudoers = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = sudoers
SPEC.loader.exec_module(sudoers)

UUID = "a3007590-0000-1000-800f-a29c1cca1511"


@pytest.fixture(autouse=True)
def host_tool_default(monkeypatch):
    monkeypatch.delenv(ENV_HOST_AMD_SMI, raising=False)


def test_render_sudoers_covers_only_exact_amd_smi_clock_commands():
    content = sudoers.render_sudoers(
        user="runner",
        amd_smi="/usr/bin/amd-smi",
        gpu_uuid=UUID,
    )

    assert "runner ALL=(root) NOPASSWD:" in content
    for expected in (
        f"/usr/bin/amd-smi set -l STABLE_PEAK -g {UUID}",
        f"/usr/bin/amd-smi set -l AUTO -g {UUID}",
    ):
        assert expected in content
    assert "rocm-smi" not in content
    assert " *" not in content


@pytest.mark.parametrize(
    ("user", "amd_smi", "label"),
    [
        ("runner\nrunner ALL=(root) NOPASSWD: ALL", "/opt/amd-smi", "safe"),
        ("runner", "/opt/amd smi", "safe"),
        ("runner", "/opt/amd-smi,ALL", "safe"),
        ("runner", "/opt/amd-smi:ALL", "safe"),
        ("runner", "/opt/amd-smi#comment", "safe"),
        ("runner", "/opt/amd-smi", "safe\nrunner ALL=(root) NOPASSWD: ALL"),
    ],
)
def test_render_sudoers_rejects_injection(user, amd_smi, label):
    with pytest.raises(ValueError, match="unsafe|absolute path"):
        sudoers.render_sudoers(user=user, amd_smi=amd_smi, label=label)


def test_default_target_user_prefers_sudo_user(monkeypatch):
    monkeypatch.setenv("SUDO_USER", "guohao")

    assert sudoers.default_target_user() == "guohao"


def test_default_target_user_falls_back_to_current_user(monkeypatch):
    monkeypatch.delenv("SUDO_USER", raising=False)
    monkeypatch.setattr(sudoers.getpass, "getuser", lambda: "runner")

    assert sudoers.default_target_user() == "runner"


def test_check_passwordless_coverage_is_read_only():
    ok = MagicMock(returncode=0, stderr="")
    with patch.object(sudoers.subprocess, "run", return_value=ok) as run:
        checks = sudoers.check_passwordless_coverage("/usr/bin/amd-smi", UUID)

    assert {check.status for check in checks} == {
        "allowed_authentication_unverified"
    }
    assert [call.args[0] for call in run.call_args_list] == [
        [
            "sudo",
            "-n",
            "-l",
            "--",
            "/usr/bin/amd-smi",
            "set",
            "-l",
            "STABLE_PEAK",
            "-g",
            UUID,
        ],
        [
            "sudo",
            "-n",
            "-l",
            "--",
            "/usr/bin/amd-smi",
            "set",
            "-l",
            "AUTO",
            "-g",
            UUID,
        ],
    ]


def test_check_passwordless_coverage_detects_password_prompt():
    failed = MagicMock(returncode=1, stderr="sudo: a password is required\n")
    with patch.object(sudoers.subprocess, "run", return_value=failed):
        checks = sudoers.check_passwordless_coverage("/usr/bin/amd-smi", UUID)

    assert {check.status for check in checks} == {"password_required"}


def test_verify_live_always_restores_auto_after_lock_failure():
    responses = [
        MagicMock(returncode=1, stderr="lock failed"),
        MagicMock(returncode=0, stderr=""),
    ]
    with patch.object(sudoers.subprocess, "run", side_effect=responses) as run:
        checks = sudoers.verify_passwordless_coverage_live(
            "/usr/bin/amd-smi",
            UUID,
        )

    assert [check.status for check in checks] == [
        "missing_or_failed",
        "covered",
    ]
    assert run.call_args_list[-1].args[0][-5:] == [
        "set",
        "-l",
        "AUTO",
        "-g",
        UUID,
    ]


def test_validate_sudoers_content_fails_closed_without_visudo(monkeypatch):
    monkeypatch.setattr(sudoers.shutil, "which", lambda _tool: None)

    with pytest.raises(FileNotFoundError, match="visudo is required"):
        sudoers.validate_sudoers_content("runner ALL=(root) /bin/true\n")


def test_install_requires_root(tmp_path):
    target = tmp_path / "sudoers"
    with (
        patch.object(sudoers.os, "geteuid", return_value=1000),
        pytest.raises(PermissionError),
    ):
        sudoers.install_sudoers("runner ALL=(root) /bin/true\n", target)


def test_install_atomically_validates_and_writes_0440_file(tmp_path):
    target = tmp_path / "sudoers"
    content = "runner ALL=(root) NOPASSWD: /opt/rocm/bin/amd-smi version\n"
    with (
        patch.object(sudoers.os, "geteuid", return_value=0),
        patch.object(sudoers, "validate_install_destination"),
        patch.object(sudoers, "validate_sudoers_content") as validate,
        patch.object(sudoers, "_visudo", return_value="/usr/sbin/visudo"),
        patch.object(sudoers.subprocess, "run", return_value=MagicMock()),
    ):
        sudoers.install_sudoers(content, target)

    assert target.read_text() == content
    assert stat_mode(target) == 0o440
    validate.assert_called_once_with(content)


def test_install_restores_previous_file_when_final_visudo_fails(tmp_path):
    target = tmp_path / "sudoers"
    target.write_text("previous\n")
    target.chmod(0o440)
    with (
        patch.object(sudoers.os, "geteuid", return_value=0),
        patch.object(sudoers, "validate_install_destination"),
        patch.object(sudoers, "validate_sudoers_content"),
        patch.object(sudoers, "_visudo", return_value="/usr/sbin/visudo"),
        patch.object(
            sudoers.subprocess,
            "run",
            side_effect=subprocess.CalledProcessError(1, "visudo"),
        ),
        pytest.raises(subprocess.CalledProcessError),
    ):
        sudoers.install_sudoers("replacement\n", target)

    assert target.read_text() == "previous\n"
    assert stat_mode(target) == 0o440


def test_install_destination_rejects_symlink(tmp_path, monkeypatch):
    monkeypatch.setattr(sudoers, "SUDOERS_DIR", tmp_path)
    monkeypatch.setattr(sudoers, "SUDOERS_PATH", tmp_path / "sudoers")
    actual = tmp_path / "actual"
    actual.write_text("unchanged\n")
    target = tmp_path / "sudoers"
    target.symlink_to(actual)

    with pytest.raises(ValueError, match="symlink"):
        sudoers.validate_install_destination(target)


def test_help_explains_check_and_live_side_effects(capsys):
    with pytest.raises(SystemExit) as exc:
        sudoers.parse_args(["--help"])

    assert exc.value.code == 0
    help_text = capsys.readouterr().out
    assert "check        Read-only" in help_text
    assert "verify-live" in help_text
    assert "only two exact device-scoped setters" in help_text
    assert "always attempts AUTO cleanup" in help_text


def stat_mode(path: Path) -> int:
    return path.stat().st_mode & 0o777


@pytest.mark.parametrize(
    "executable", ["/usr/bin/amd-smi", "/opt/rocm/bin/amd-smi"]
)
@pytest.mark.parametrize("gpu_uuid", [UUID])
def test_runtime_and_sudoers_share_exact_clock_contract(executable, gpu_uuid):
    from sol_execbench.core.platform.clock_commands import (
        ClockLevel,
        clock_command,
    )

    content = sudoers.render_sudoers(
        user="runner", amd_smi=executable, gpu_uuid=gpu_uuid
    )
    commands = sudoers.protected_commands(executable, gpu_uuid)
    for level in ClockLevel:
        command = clock_command(executable, level, gpu_uuid)
        assert command in commands
        assert " ".join(command) in content


@pytest.mark.parametrize("gpu_uuid", ["all", "0", "*", "bad\nALL", "-g other"])
def test_uuid_rule_rejects_unverified_selectors(gpu_uuid):
    with pytest.raises(ValueError, match="UUID"):
        sudoers.render_sudoers(
            user="runner", amd_smi="/opt/rocm/bin/amd-smi", gpu_uuid=gpu_uuid
        )


def test_default_rule_path_does_not_follow_installers_path(monkeypatch):
    from sol_execbench.core.platform import runtime

    monkeypatch.setattr(
        runtime,
        "resolve_rocm_tool_command",
        lambda _tool: "/opt/rocm/core-10.0/bin/amd-smi",
    )
    assert sudoers.resolve_amd_smi() == "/usr/bin/amd-smi"


def test_explicit_symlink_spelling_is_not_resolved(tmp_path):
    target = tmp_path / "versioned-tool"
    target.write_text("fixture only")
    link = tmp_path / "amd-smi"
    link.symlink_to(target)
    assert sudoers.resolve_amd_smi(str(link)) == str(link)
    text = sudoers.render_sudoers(
        user="guohao", amd_smi=str(link), gpu_uuid=UUID
    )
    assert str(target) not in text


def test_missing_uuid_is_rejected_by_renderer_and_cli():
    with pytest.raises(ValueError, match="gpu_uuid is required"):
        sudoers.render_sudoers(user="guohao", amd_smi="/usr/bin/amd-smi")
    with pytest.raises(SystemExit) as error:
        sudoers.parse_args(["--mode", "print", "--user", "guohao"])
    assert error.value.code == 2


def test_local_rule_contains_exactly_two_target_setters():
    text = sudoers.render_sudoers(
        user="guohao", amd_smi=sudoers.resolve_amd_smi(), gpu_uuid=UUID
    )
    assert sudoers.protected_commands("/usr/bin/amd-smi", UUID) == [
        ["/usr/bin/amd-smi", "set", "-l", "STABLE_PEAK", "-g", UUID],
        ["/usr/bin/amd-smi", "set", "-l", "AUTO", "-g", UUID],
    ]
    assert text.count(UUID) == 2
    assert "version" not in text and "/opt/" not in text
    assert "NOPASSWD: ALL" not in text and "*" not in text


@pytest.mark.parametrize(
    "selected",
    [
        "/usr/bin/amd-smi",
        "/opt/rocm/bin/amd-smi",
        "/opt/rocm/core-10.0/bin/amd-smi",
    ],
    ids=["system-first", "rocm-first", "versioned-first"],
)
@pytest.mark.parametrize(
    "override",
    [None, "/fixture/amd-smi"],
    ids=["default", "explicit-host-path"],
)
def test_host_setters_and_default_rule_ignore_path_order(
    monkeypatch, selected, override
):
    from sol_execbench.core.bench import clock_lock
    from sol_execbench.core.platform import runtime

    resolve = runtime.resolve_rocm_tool_command
    monkeypatch.setattr(
        runtime,
        "resolve_rocm_tool_command",
        lambda tool: resolve(tool, which=lambda _: selected),
    )
    if override is not None:
        monkeypatch.setenv(ENV_HOST_AMD_SMI, override)
    monkeypatch.setattr(clock_lock.time, "sleep", lambda _: None)
    calls = []
    levels = []

    def run(command, **_):
        calls.append(command)
        if command[0] == "sudo":
            if command[2] != "-l":
                levels.append(command[5])
            return subprocess.CompletedProcess(command, 0, "", "")
        return subprocess.CompletedProcess(
            command,
            0,
            '{"gpu_data":[{"gpu":0,"perf_level":"AMDSMI_DEV_PERF_LEVEL_'
            + levels[-1]
            + '"}]}',
            "",
        )

    monkeypatch.setattr(clock_lock.subprocess, "run", run)
    clock_lock.set_peak_clocks(UUID)
    assert clock_lock.unlock_clocks(UUID)
    tool = sudoers.resolve_amd_smi()
    sudoers.check_passwordless_coverage(tool, UUID)
    expected = sudoers.protected_commands(tool, UUID)
    setters = [
        cmd[2:] for cmd in calls if cmd[:2] == ["sudo", "-n"] and cmd[2] != "-l"
    ]
    assert tool == (override or "/usr/bin/amd-smi")
    assert setters == expected
    assert all(cmd[0] == tool for cmd in calls if cmd[0] != "sudo")
    assert calls[-2:] == [["sudo", "-n", "-l", "--", *cmd] for cmd in expected]


def test_explicit_host_selection_preserves_cli_and_symlink_spelling(
    tmp_path, monkeypatch
):
    from sol_execbench.core.platform.clock_commands import host_clock_executable

    target = tmp_path / "versioned-tool"
    target.write_text("CPU fixture")
    link = tmp_path / "amd-smi"
    link.symlink_to(target)
    monkeypatch.setenv(ENV_HOST_AMD_SMI, str(link))
    assert host_clock_executable() == sudoers.resolve_amd_smi() == str(link)
    args = sudoers.parse_args(
        [
            "--mode",
            "print",
            "--gpu-uuid",
            UUID,
            "--amd-smi",
            "/explicit/amd-smi",
        ]
    )
    assert (
        sudoers.resolve_amd_smi(args.amd_smi)
        == host_clock_executable(args.amd_smi)
        == "/explicit/amd-smi"
    )


@pytest.mark.parametrize(
    "configured", ["", "amd-smi", "relative/amd-smi", "/bad\0tool"]
)
def test_invalid_host_configuration_is_not_replaced_by_path(
    monkeypatch, configured
):
    from sol_execbench.core.platform.clock_commands import host_clock_executable

    if "\0" not in configured:
        monkeypatch.setenv(ENV_HOST_AMD_SMI, configured)
    with pytest.raises(ValueError, match="absolute path"):
        host_clock_executable(configured)
    if "\0" not in configured:
        with pytest.raises(ValueError, match="absolute path"):
            sudoers.resolve_amd_smi()


def test_cli_reports_invalid_host_configuration_without_execution(
    monkeypatch, capsys
):
    monkeypatch.setenv(ENV_HOST_AMD_SMI, "relative/amd-smi")
    with patch.object(sudoers.subprocess, "run") as run:
        assert (
            sudoers.main(["--mode", "check", "--gpu-uuid", UUID, "--json"]) == 1
        )
    assert json.loads(capsys.readouterr().out)["status"] == "failed"
    run.assert_not_called()


def test_legacy_tool_discovery_is_independent_of_host_override(monkeypatch):
    from sol_execbench.core.platform import runtime
    from sol_execbench.core.platform.clock_commands import clock_executable

    resolve = runtime.resolve_rocm_tool_command
    monkeypatch.setattr(
        runtime,
        "resolve_rocm_tool_command",
        lambda tool: resolve(tool, which=lambda _: "/legacy/amd-smi"),
    )
    monkeypatch.setenv(ENV_HOST_AMD_SMI, "/host/amd-smi")
    assert clock_executable() == "/legacy/amd-smi"
    assert clock_executable("/explicit/amd-smi") == "/explicit/amd-smi"
