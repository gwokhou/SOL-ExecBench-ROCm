# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
"""Install or check passwordless sudoers coverage for ROCm clock commands."""

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


SUDOERS_PATH = Path("/etc/sudoers.d/sol-execbench-rocm-smi")
DEFAULT_LABEL = "sol-execbench-rocm-smi"


@dataclass(frozen=True)
class SudoCommandCheck:
    command: list[str]
    status: str
    returncode: int | None
    stderr_tail: str


def resolve_rocm_smi(explicit_path: str | None = None) -> str:
    if explicit_path:
        return explicit_path
    return shutil.which("rocm-smi") or "/opt/rocm/bin/rocm-smi"


def default_target_user() -> str:
    """Return the user that should receive sudoers coverage."""
    return os.environ.get("SUDO_USER") or getpass.getuser()


def sudoers_command_patterns(rocm_smi: str) -> list[str]:
    return [
        f"{rocm_smi} --showclocks",
        f"{rocm_smi} -s",
        f"{rocm_smi} --showperflevel",
        f"{rocm_smi} --showclkfrq",
        f"{rocm_smi} --setperflevel manual",
        f"{rocm_smi} --setperflevel auto",
        f"{rocm_smi} --setsclk *",
        f"{rocm_smi} --setmclk *",
        f"{rocm_smi} --resetclocks",
    ]


def render_sudoers(
    *,
    user: str,
    rocm_smi: str,
    label: str = DEFAULT_LABEL,
) -> str:
    patterns = ", \\\n    ".join(sudoers_command_patterns(rocm_smi))
    return (
        f"# Managed by SOL ExecBench ROCm: {label}\n"
        f"# Allows only ROCm SMI clock query, lock, and reset commands.\n"
        f"{user} ALL=(root) NOPASSWD: {patterns}\n"
    )


def check_commands(rocm_smi: str) -> list[list[str]]:
    return [
        ["sudo", "-n", rocm_smi, "--showclocks"],
        ["sudo", "-n", rocm_smi, "-s"],
        ["sudo", "-n", rocm_smi, "--showperflevel"],
        ["sudo", "-n", rocm_smi, "--showclkfrq"],
        ["sudo", "-n", rocm_smi, "--setperflevel", "manual"],
        ["sudo", "-n", rocm_smi, "--setperflevel", "auto"],
        ["sudo", "-n", rocm_smi, "--setsclk", "0"],
        ["sudo", "-n", rocm_smi, "--setmclk", "0"],
        ["sudo", "-n", rocm_smi, "--resetclocks"],
    ]


def _tail(value: str, limit: int = 400) -> str:
    return value[-limit:] if len(value) > limit else value


def check_passwordless_coverage(rocm_smi: str) -> list[SudoCommandCheck]:
    results: list[SudoCommandCheck] = []
    for command in check_commands(rocm_smi):
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
    parser.add_argument("--rocm-smi", default=None)
    parser.add_argument("--sudoers-path", type=Path, default=SUDOERS_PATH)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rocm_smi = resolve_rocm_smi(args.rocm_smi)
    content = render_sudoers(user=args.user, rocm_smi=rocm_smi)

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

    checks = check_passwordless_coverage(rocm_smi)
    all_covered = all(check.status == "covered" for check in checks)
    payload = {
        "status": "covered" if all_covered else "missing",
        "rocm_smi": rocm_smi,
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
