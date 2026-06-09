# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
"""Install or check passwordless sudoers coverage for ROCm clock commands.

Covers the ``amd-smi`` commands used by SOL ExecBench clock locking:
``version`` for passwordless probing, ``set -l STABLE_PEAK`` for locking, and
``set -l AUTO`` for cleanup.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path


SUDOERS_PATH = Path("/etc/sudoers.d/sol-execbench-amd-smi")
DEFAULT_LABEL = "sol-execbench-amd-smi"


@dataclass(frozen=True)
class SudoCommandCheck:
    command: list[str]
    status: str
    returncode: int | None
    stderr_tail: str


def resolve_amd_smi(explicit_path: str | None = None) -> str:
    """Return the path to amd-smi."""
    if explicit_path:
        return explicit_path
    return shutil.which("amd-smi") or "/opt/rocm/bin/amd-smi"


def default_target_user() -> str:
    """Return the user that should receive sudoers coverage."""
    return os.environ.get("SUDO_USER") or getpass.getuser()


def amd_smi_command_patterns(amd_smi: str) -> list[str]:
    return [
        f"{amd_smi} version",
        f"{amd_smi} set -l STABLE_PEAK",
        f"{amd_smi} set -l AUTO",
    ]


def render_sudoers(
    *,
    user: str,
    amd_smi: str,
    label: str = DEFAULT_LABEL,
) -> str:
    patterns = ", \\\n    ".join(amd_smi_command_patterns(amd_smi))
    lines = [
        f"# Managed by SOL ExecBench ROCm: {label}",
        "# Allows only amd-smi clock probe, lock, and reset commands.",
        f"{user} ALL=(root) NOPASSWD: {patterns}",
    ]
    return "\n".join(lines) + "\n"


def check_commands(amd_smi: str) -> list[list[str]]:
    return [
        ["sudo", "-n", amd_smi, "version"],
        ["sudo", "-n", amd_smi, "set", "-l", "STABLE_PEAK"],
        ["sudo", "-n", amd_smi, "set", "-l", "AUTO"],
    ]


def _tail(value: str, limit: int = 400) -> str:
    return value[-limit:] if len(value) > limit else value


def check_passwordless_coverage(amd_smi: str) -> list[SudoCommandCheck]:
    results: list[SudoCommandCheck] = []
    for command in check_commands(amd_smi):
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
            )
        except FileNotFoundError as exc:
            results.append(
                SudoCommandCheck(
                    command=command,
                    status="missing_tool",
                    returncode=None,
                    stderr_tail=str(exc),
                )
            )
            continue
        stderr = result.stderr or ""
        status = "covered" if result.returncode == 0 else "missing_or_failed"
        if "password" in stderr.lower() or "a password is required" in stderr.lower():
            status = "password_required"
        results.append(
            SudoCommandCheck(
                command=command,
                status=status,
                returncode=result.returncode,
                stderr_tail=_tail(stderr.strip()),
            )
        )
    return results


def validate_sudoers_content(content: str) -> None:
    visudo = shutil.which("visudo")
    if visudo is None:
        return
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as handle:
        handle.write(content)
        temp_path = Path(handle.name)
    try:
        subprocess.run([visudo, "-cf", str(temp_path)], check=True)
    finally:
        temp_path.unlink(missing_ok=True)


def install_sudoers(content: str, path: Path = SUDOERS_PATH) -> None:
    if os.geteuid() != 0:
        raise PermissionError("install requires root; rerun with sudo")
    validate_sudoers_content(content)
    path.write_text(content, encoding="utf-8")
    path.chmod(0o440)
    validate_sudoers_content(path.read_text(encoding="utf-8"))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["print", "check", "install"],
        default="check",
    )
    parser.add_argument("--user", default=default_target_user())
    parser.add_argument("--amd-smi", default=None)
    parser.add_argument("--sudoers-path", type=Path, default=SUDOERS_PATH)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    amd_smi = resolve_amd_smi(args.amd_smi)
    content = render_sudoers(user=args.user, amd_smi=amd_smi)

    if args.mode == "print":
        print(content, end="")
        return 0

    if args.mode == "install":
        try:
            install_sudoers(content, args.sudoers_path)
        except Exception as exc:
            payload = {"status": "failed", "error": str(exc)}
            if args.json:
                print(json.dumps(payload, sort_keys=True))
            else:
                print(f"failed: {exc}", file=sys.stderr)
            return 1

    checks = check_passwordless_coverage(amd_smi)
    all_covered = all(check.status == "covered" for check in checks)
    payload = {
        "status": "covered" if all_covered else "missing",
        "amd_smi": amd_smi,
        "checks": [asdict(check) for check in checks],
    }
    if args.json:
        print(json.dumps(payload, sort_keys=True))
    else:
        for check in checks:
            print(f"{check.status}: {' '.join(check.command)}")
    return 0 if all_covered else 1


if __name__ == "__main__":
    raise SystemExit(main())
