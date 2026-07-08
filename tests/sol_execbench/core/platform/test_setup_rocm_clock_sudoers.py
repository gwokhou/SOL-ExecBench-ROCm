from __future__ import annotations

import importlib.util
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


def test_render_sudoers_covers_amd_smi_clock_commands():
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


def test_render_sudoers_excludes_rocm_smi_commands():
    content = sudoers.render_sudoers(
        user="runner",
        amd_smi="/opt/rocm/bin/amd-smi",
    )

    assert "rocm-smi" not in content


def test_default_target_user_prefers_sudo_user(monkeypatch):
    monkeypatch.setenv("SUDO_USER", "guohao")

    assert sudoers.default_target_user() == "guohao"


def test_default_target_user_falls_back_to_current_user(monkeypatch):
    monkeypatch.delenv("SUDO_USER", raising=False)
    monkeypatch.setattr(sudoers.getpass, "getuser", lambda: "runner")

    assert sudoers.default_target_user() == "runner"


def test_check_passwordless_coverage_reports_all_covered():
    ok = MagicMock(returncode=0, stderr="")
    with patch.object(sudoers.subprocess, "run", return_value=ok):
        checks = sudoers.check_passwordless_coverage("/opt/rocm/bin/amd-smi")

    assert {check.status for check in checks} == {"covered"}
    assert checks[0].command == ["sudo", "-n", "/opt/rocm/bin/amd-smi", "version"]


def test_check_passwordless_coverage_detects_password_prompt():
    failed = MagicMock(returncode=1, stderr="sudo: a password is required\n")
    with patch.object(sudoers.subprocess, "run", return_value=failed):
        checks = sudoers.check_passwordless_coverage("/opt/rocm/bin/amd-smi")

    assert {check.status for check in checks} == {"password_required"}


def test_check_passwordless_coverage_checks_lock_and_unlock():
    ok = MagicMock(returncode=0, stderr="")
    with patch.object(sudoers.subprocess, "run", return_value=ok):
        checks = sudoers.check_passwordless_coverage("/usr/bin/amd-smi")

    assert {check.status for check in checks} == {"covered"}
    commands_str = [" ".join(c.command) for c in checks]
    assert "sudo -n /usr/bin/amd-smi set -l STABLE_PEAK" in commands_str
    assert "sudo -n /usr/bin/amd-smi set -l AUTO" in commands_str


def test_install_requires_root(tmp_path):
    target = tmp_path / "sudoers"
    with patch.object(sudoers.os, "geteuid", return_value=1000):
        with pytest.raises(PermissionError):
            sudoers.install_sudoers("runner ALL=(root) NOPASSWD: /bin/true\n", target)


def test_install_validates_and_writes_file(tmp_path):
    target = tmp_path / "sudoers"
    content = "runner ALL=(root) NOPASSWD: /opt/rocm/bin/amd-smi set -l STABLE_PEAK\n"
    with (
        patch.object(sudoers.os, "geteuid", return_value=0),
        patch.object(sudoers, "validate_sudoers_content") as validate,
    ):
        sudoers.install_sudoers(content, target)

    assert target.read_text() == content
    assert oct(target.stat().st_mode & 0o777) == "0o440"
    assert validate.call_count == 2
