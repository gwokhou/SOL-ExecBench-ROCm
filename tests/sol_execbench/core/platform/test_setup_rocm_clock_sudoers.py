from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

SCRIPT_PATH = (
    Path(__file__).resolve().parents[4] / "scripts/setup_rocm_clock_sudoers.py"
)
SPEC = importlib.util.spec_from_file_location("setup_rocm_clock_sudoers", SCRIPT_PATH)
assert SPEC is not None
assert SPEC.loader is not None
sudoers = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = sudoers
SPEC.loader.exec_module(sudoers)


def test_render_sudoers_covers_only_exact_amd_smi_clock_commands():
    content = sudoers.render_sudoers(
        user="runner",
        amd_smi="/opt/rocm/bin/amd-smi",
    )

    assert "runner ALL=(root) NOPASSWD:" in content
    for expected in (
        "/opt/rocm/bin/amd-smi version",
        "/opt/rocm/bin/amd-smi set -l STABLE_PEAK",
        "/opt/rocm/bin/amd-smi set -l AUTO",
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
    with pytest.raises(ValueError):
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
        checks = sudoers.check_passwordless_coverage("/opt/rocm/bin/amd-smi")

    assert {check.status for check in checks} == {"covered"}
    assert [call.args[0] for call in run.call_args_list] == [
        ["sudo", "-n", "-l", "--", "/opt/rocm/bin/amd-smi", "version"],
        [
            "sudo",
            "-n",
            "-l",
            "--",
            "/opt/rocm/bin/amd-smi",
            "set",
            "-l",
            "STABLE_PEAK",
        ],
        [
            "sudo",
            "-n",
            "-l",
            "--",
            "/opt/rocm/bin/amd-smi",
            "set",
            "-l",
            "AUTO",
        ],
    ]


def test_check_passwordless_coverage_detects_password_prompt():
    failed = MagicMock(returncode=1, stderr="sudo: a password is required\n")
    with patch.object(sudoers.subprocess, "run", return_value=failed):
        checks = sudoers.check_passwordless_coverage("/opt/rocm/bin/amd-smi")

    assert {check.status for check in checks} == {"password_required"}


def test_verify_live_always_restores_auto_after_lock_failure():
    responses = [
        MagicMock(returncode=0, stderr=""),
        MagicMock(returncode=1, stderr="lock failed"),
        MagicMock(returncode=0, stderr=""),
    ]
    with patch.object(sudoers.subprocess, "run", side_effect=responses) as run:
        checks = sudoers.verify_passwordless_coverage_live("/opt/rocm/bin/amd-smi")

    assert [check.status for check in checks] == [
        "covered",
        "missing_or_failed",
        "covered",
    ]
    assert run.call_args_list[-1].args[0][-3:] == ["set", "-l", "AUTO"]


def test_validate_sudoers_content_fails_closed_without_visudo(monkeypatch):
    monkeypatch.setattr(sudoers.shutil, "which", lambda _tool: None)

    with pytest.raises(FileNotFoundError, match="visudo is required"):
        sudoers.validate_sudoers_content("runner ALL=(root) /bin/true\n")


def test_install_requires_root(tmp_path):
    target = tmp_path / "sudoers"
    with patch.object(sudoers.os, "geteuid", return_value=1000):
        with pytest.raises(PermissionError):
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
    assert "every visible AMD GPU" in help_text
    assert "always attempts AUTO cleanup" in help_text


def stat_mode(path: Path) -> int:
    return path.stat().st_mode & 0o777
