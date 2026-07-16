# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Build and locate the project-owned C++ JSON helper on demand."""

from __future__ import annotations

import hashlib
from importlib import resources
import os
from pathlib import Path
import platform
import subprocess

from sol_execbench.core.process.subprocesses import run_in_process_group
from sol_execbench.tools.amd_isa.errors import IsaHelperBuildError
from sol_execbench.tools.amd_isa.repository import _cache_root, _file_lock


def _resource_path(relative: str) -> Path:
    return Path(str(resources.files("sol_execbench.tools.amd_isa").joinpath(relative)))


def _build_key() -> str:
    digest = hashlib.sha256(
        b"isa_spec_manager-v1.2.0-507111dea876c3d4d268c1c86344835446791e41"
    )
    root = _resource_path("native")
    vendor = _resource_path("../../_vendor/isa_spec_manager")
    for base in (root, vendor):
        for path in sorted(base.rglob("*")):
            if path.is_file():
                digest.update(path.relative_to(base).as_posix().encode())
                digest.update(path.read_bytes())
    digest.update(platform.platform().encode())
    return digest.hexdigest()[:24]


def helper_path(cache_root: Path | None = None) -> Path:
    root = cache_root or _cache_root()
    return (
        root
        / "helper"
        / _build_key()
        / "build"
        / "bin"
        / "sol-execbench-amd-isa-helper"
    )


def ensure_helper(cache_root: Path | None = None) -> Path:
    executable = helper_path(cache_root)
    if executable.is_file() and os.access(executable, os.X_OK):
        return executable
    build_dir = executable.parent.parent
    with _file_lock(build_dir.parent / ".lock"):
        if executable.is_file() and os.access(executable, os.X_OK):
            return executable
        source_dir = _resource_path("native")
        try:
            configure = run_in_process_group(
                [
                    "cmake",
                    "-S",
                    str(source_dir),
                    "-B",
                    str(build_dir),
                    "-DCMAKE_BUILD_TYPE=Release",
                ],
                timeout=120,
            )
            if configure.returncode != 0:
                raise IsaHelperBuildError(
                    (configure.stderr or configure.stdout)[-4000:]
                )
            build = run_in_process_group(
                ["cmake", "--build", str(build_dir), "--parallel"],
                timeout=300,
            )
            if build.returncode != 0:
                raise IsaHelperBuildError((build.stderr or build.stdout)[-4000:])
        except FileNotFoundError as exc:
            raise IsaHelperBuildError(
                "cmake and a C++14 compiler are required for AMD ISA support"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise IsaHelperBuildError("AMD ISA helper build timed out") from exc
        if not executable.is_file() or not os.access(executable, os.X_OK):
            raise IsaHelperBuildError(
                "AMD ISA helper build finished without an executable"
            )
    return executable
