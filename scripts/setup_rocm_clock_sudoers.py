# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
"""Safely install and verify passwordless sudoers coverage for ROCm clocks."""

from __future__ import annotations

import argparse
import contextlib
import getpass
import json
import os
import pwd
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

from sol_execbench.core.platform.clock_commands import (
    ClockLevel,
    clock_command,
    host_clock_executable,
)

SUDOERS_PATH = Path("/etc/sudoers.d/sol-execbench-amd-smi")
SUDOERS_DIR = Path("/etc/sudoers.d")
DEFAULT_LABEL = "sol-execbench-amd-smi"
COMMAND_TIMEOUT_S = 30
_USER_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_.-]{0,31}\$?")
_LABEL_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")
_ABSOLUTE_PATH_PATTERN = re.compile(r"/(?:[A-Za-z0-9._+-]+/)*[A-Za-z0-9._+-]+")
_COMMAND_FAILURE_MARKERS = ("unable to set", "failed to set", "error:")

HELP_EPILOG = """\
Modes:
  check        Read-only. Check allowance; success does not prove NOPASSWD.
  verify-live  Explicitly enter STABLE_PEAK, then restore AUTO in a finally block.
  print        Print the validated sudoers entry without installing it.
  install      Atomically install a visudo-validated 0440 sudoers file as root.

The rule requires --gpu-uuid and covers only two exact device-scoped setters.
The default invocation is the stable /usr/bin/amd-smi link, independent of
PATH order. SOL_EXECBENCH_HOST_AMD_SMI explicitly selects another absolute
host path for both UUID clock control and this tool. --amd-smi overrides it
for this invocation only. Link resolution validates safety, not rule rendering.

Typical workflow (run from the repository root):
  python scripts/setup_rocm_clock_sudoers.py --mode print \\
    --user guohao --gpu-uuid a3007590-0000-1000-800f-a29c1cca1511
  sudo .venv/bin/python scripts/setup_rocm_clock_sudoers.py --mode install \\
    --user guohao --gpu-uuid a3007590-0000-1000-800f-a29c1cca1511
  .venv/bin/python scripts/setup_rocm_clock_sudoers.py --mode check --json \\
    --gpu-uuid a3007590-0000-1000-800f-a29c1cca1511

Use `verify-live` only with explicit approval to change the selected GPU.
It always attempts AUTO cleanup, including when the lock command fails. The
installer refuses unsafe usernames, labels, executable paths, symlink targets,
non-root-controlled executables, and destinations outside /etc/sudoers.d.

Known legacy cleanup:
  This script does not delete older rules automatically. If `sudo -l` still
  lists dead `/usr/bin/rocm-smi` rules or `/usr/bin/amd-smi set -l *`, first
  confirm no other workload depends on these files, then move them outside the
  parsed sudoers directory and validate the complete policy:

    sudo install -d -m 0700 /var/backups/sol-execbench-sudoers
    sudo mv /etc/sudoers.d/sol-execbench-rocm-smi \\
      /var/backups/sol-execbench-sudoers/
    sudo mv /etc/sudoers.d/hip-bench-gpu-perf \\
      /var/backups/sol-execbench-sudoers/
    sudo visudo -c

  To restore them, move the exact files back, set mode 0440, and run
  `sudo visudo -c` again. Never remove the active sol-execbench-amd-smi rule
  while benchmark clock locking or the gfx1200 profiler wrapper uses it.
"""


@dataclass(frozen=True)
class SudoCommandCheck:
    """Result of one passwordless sudo policy check."""

    command: list[str]
    status: str
    returncode: int | None
    stderr_tail: str


def resolve_amd_smi(explicit_path: str | None = None) -> str:
    """Preserve the stable invocation spelling, regardless of installer PATH."""
    return host_clock_executable(explicit_path)


def default_target_user() -> str:
    """Return the user that should receive sudoers coverage."""
    return os.environ.get("SUDO_USER") or getpass.getuser()


def validate_render_inputs(*, user: str, amd_smi: str, label: str) -> None:
    """Reject values that could alter sudoers token or line boundaries."""
    if _USER_PATTERN.fullmatch(user) is None:
        raise ValueError(f"unsafe sudoers user: {user!r}")
    if _LABEL_PATTERN.fullmatch(label) is None:
        raise ValueError(f"unsafe sudoers label: {label!r}")
    if _ABSOLUTE_PATH_PATTERN.fullmatch(amd_smi) is None:
        raise ValueError(
            "amd-smi must be an absolute path containing only safe sudoers characters",
        )


def validate_target_user(user: str) -> None:
    """Require the sudoers target to be an existing local account."""
    try:
        pwd.getpwnam(user)
    except KeyError as exc:
        raise ValueError(f"target user does not exist: {user}") from exc


def _assert_root_controlled(
    path: Path,
    *,
    follow_symlinks: bool = True,
) -> None:
    info = path.stat(follow_symlinks=follow_symlinks)
    permissions_are_unsafe = (
        not stat.S_ISLNK(info.st_mode) and info.st_mode & 0o022
    )
    if info.st_uid != 0 or permissions_are_unsafe:
        raise PermissionError(f"path is not root-controlled: {path}")


def validate_amd_smi_executable(amd_smi: str) -> None:
    """Require an executable whose path and target are controlled by root."""
    path = Path(amd_smi)
    if not path.is_file() or not os.access(path, os.X_OK):
        raise FileNotFoundError(f"amd-smi is not executable: {path}")
    for candidate in (path, *path.parents):
        _assert_root_controlled(candidate, follow_symlinks=False)
    for candidate in (path.resolve(), *path.resolve().parents):
        _assert_root_controlled(candidate)


def amd_smi_command_patterns(
    amd_smi: str, gpu_uuid: str | None = None
) -> list[str]:
    """Return the exact AMD SMI command patterns allowed by sudoers."""
    return [
        " ".join(command) for command in protected_commands(amd_smi, gpu_uuid)
    ]


def render_sudoers(
    *,
    user: str,
    amd_smi: str,
    label: str = DEFAULT_LABEL,
    gpu_uuid: str | None = None,
) -> str:
    """Render a narrowly scoped sudoers rule for AMD SMI clock control."""
    validate_render_inputs(user=user, amd_smi=amd_smi, label=label)
    patterns = ", \\\n    ".join(amd_smi_command_patterns(amd_smi, gpu_uuid))
    lines = [
        f"# Managed by SOL ExecBench ROCm: {label}",
        "# Allows only device-scoped STABLE_PEAK and AUTO setters.",
        f"{user} ALL=(root) NOPASSWD: {patterns}",
    ]
    return "\n".join(lines) + "\n"


def protected_commands(
    amd_smi: str, gpu_uuid: str | None = None
) -> list[list[str]]:
    """Return the clock commands protected by the sudoers rule."""
    if gpu_uuid is None:
        raise ValueError(
            "gpu_uuid is required; all-device rules are not generated"
        )
    return [
        clock_command(amd_smi, ClockLevel.PEAK, gpu_uuid),
        clock_command(amd_smi, ClockLevel.AUTO, gpu_uuid),
    ]


def check_commands(
    amd_smi: str, gpu_uuid: str | None = None
) -> list[list[str]]:
    """Return read-only sudo policy queries, never the protected commands."""
    return [
        ["sudo", "-n", "-l", "--", *command]
        for command in protected_commands(amd_smi, gpu_uuid)
    ]


def _tail(value: str, limit: int = 400) -> str:
    return value[-limit:] if len(value) > limit else value


def _run_check(command: list[str]) -> SudoCommandCheck:
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=COMMAND_TIMEOUT_S,
        )
    except FileNotFoundError as exc:
        return SudoCommandCheck(command, "missing_tool", None, str(exc))
    except subprocess.TimeoutExpired as exc:
        return SudoCommandCheck(command, "timed_out", None, str(exc))
    stderr = result.stderr or ""
    output = f"{result.stdout or ''}\n{stderr}".lower()
    status = "covered" if result.returncode == 0 else "missing_or_failed"
    if result.returncode == 0 and command[:4] == ["sudo", "-n", "-l", "--"]:
        status = "allowed_authentication_unverified"
    if "password" in stderr.lower():
        status = "password_required"
    elif any(marker in output for marker in _COMMAND_FAILURE_MARKERS):
        status = "command_failed"
    return SudoCommandCheck(
        command,
        status,
        result.returncode,
        _tail(stderr.strip()),
    )


def check_passwordless_coverage(
    amd_smi: str, gpu_uuid: str | None = None
) -> list[SudoCommandCheck]:
    """Inspect exact-command sudo policy without changing GPU state."""
    return [
        _run_check(command) for command in check_commands(amd_smi, gpu_uuid)
    ]


def verify_passwordless_coverage_live(
    amd_smi: str, gpu_uuid: str | None = None
) -> list[SudoCommandCheck]:
    """Exercise the clock lifecycle and always attempt to restore AUTO."""
    stable_peak, auto = protected_commands(amd_smi, gpu_uuid)
    results = []
    try:
        results.append(_run_check(["sudo", "-n", *stable_peak]))
    finally:
        results.append(_run_check(["sudo", "-n", *auto]))
    return results


def _visudo() -> str:
    executable = shutil.which("visudo")
    if executable is None:
        raise FileNotFoundError(
            "visudo is required to validate sudoers content",
        )
    return executable


def validate_sudoers_content(content: str) -> None:
    """Fail closed unless visudo accepts the complete generated content."""
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        delete=False,
    ) as handle:
        handle.write(content)
        temp_path = Path(handle.name)
    try:
        subprocess.run(
            [_visudo(), "-cf", str(temp_path)],
            check=True,
            timeout=10,
        )
    finally:
        temp_path.unlink(missing_ok=True)


def validate_install_destination(path: Path) -> None:
    """Reject alternate directories, symlinks, and writable sudoers parents."""
    if path.parent.resolve() != SUDOERS_DIR or path.name != SUDOERS_PATH.name:
        raise ValueError(f"sudoers destination must be {SUDOERS_PATH}")
    if path.is_symlink():
        raise ValueError(f"refusing sudoers symlink destination: {path}")
    _assert_root_controlled(path.parent)
    if path.exists():
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode):
            raise PermissionError(
                f"existing sudoers destination is unsafe: {path}",
            )
        _assert_root_controlled(path, follow_symlinks=False)


def _stage_sudoers(path: Path, content: bytes) -> Path:
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
    )
    staged = Path(name)
    try:
        os.fchmod(descriptor, 0o440)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        with contextlib.suppress(OSError):
            os.close(descriptor)
        staged.unlink(missing_ok=True)
        raise
    return staged


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _restore_previous(path: Path, previous: bytes | None) -> None:
    if previous is None:
        path.unlink(missing_ok=True)
    else:
        restored = _stage_sudoers(path, previous)
        os.replace(restored, path)
    _fsync_directory(path.parent)


def install_sudoers(content: str, path: Path = SUDOERS_PATH) -> None:
    """Atomically install validated content and restore the old file on failure."""
    if os.geteuid() != 0:
        raise PermissionError("install requires root; rerun with sudo")
    validate_install_destination(path)
    validate_sudoers_content(content)
    previous = path.read_bytes() if path.exists() else None
    staged = _stage_sudoers(path, content.encode())
    try:
        os.replace(staged, path)
        _fsync_directory(path.parent)
        subprocess.run([_visudo(), "-cf", str(path)], check=True, timeout=10)
    except BaseException:
        staged.unlink(missing_ok=True)
        _restore_previous(path, previous)
        raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=HELP_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["print", "check", "verify-live", "install"],
        default="check",
    )
    parser.add_argument("--user", default=default_target_user())
    parser.add_argument("--amd-smi", default=None)
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--gpu-uuid",
        required=True,
        help="Verified AMD-SMI UUID; required to avoid all-device scope",
    )
    return parser.parse_args(argv)


def _print_checks(
    checks: list[SudoCommandCheck],
    amd_smi: str,
    as_json: bool,
) -> bool:
    all_covered = all(check.status == "covered" for check in checks)
    payload = {
        "status": "covered" if all_covered else "missing",
        "amd_smi": amd_smi,
        "checks": [asdict(check) for check in checks],
    }
    if as_json:
        print(json.dumps(payload, sort_keys=True))
    else:
        for check in checks:
            print(f"{check.status}: {' '.join(check.command)}")
    return all_covered


def main(argv: list[str] | None = None) -> int:
    """Validate, render, install, or check the clock sudoers rule."""
    args = parse_args(argv)
    try:
        amd_smi = resolve_amd_smi(args.amd_smi)
        validate_render_inputs(
            user=args.user,
            amd_smi=amd_smi,
            label=DEFAULT_LABEL,
        )
        validate_target_user(args.user)
        if args.mode != "print":
            validate_amd_smi_executable(amd_smi)
        content = render_sudoers(
            user=args.user, amd_smi=amd_smi, gpu_uuid=args.gpu_uuid
        )
        if args.mode == "print":
            print(content, end="")
            return 0
        if args.mode == "install":
            install_sudoers(content)
            return 0
        checks = (
            verify_passwordless_coverage_live(amd_smi, args.gpu_uuid)
            if args.mode == "verify-live"
            else check_passwordless_coverage(amd_smi, args.gpu_uuid)
        )
        return 0 if _print_checks(checks, amd_smi, args.json) else 1
    except Exception as exc:  # noqa: BLE001 -- CLI boundary reports all failures
        payload = {"status": "failed", "error": str(exc)}
        if args.json:
            print(json.dumps(payload, sort_keys=True))
        else:
            print(f"failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
