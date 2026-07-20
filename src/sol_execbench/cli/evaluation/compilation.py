# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Compilation subprocess helpers for the SOL-ExecBench CLI."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

from ...core.bench.io import flashinfer_safetensors_env
from ...core.bench.stderr import filter_benign_rocm_stderr
from ...core.process.environment import sanitized_subprocess_env
from ...core.process.subprocesses import (
    TextSubprocessRunner,
    run_in_process_group_bounded,
)


class CompilePackagerBase(Protocol):
    """Common staged-package behavior needed by the compile phase."""

    @property
    def _is_cpp(self) -> bool: ...


class NativeCompilePackager(CompilePackagerBase, Protocol):
    def compile(self) -> tuple[list[str], str]: ...


class CommandCompilePackager(CompilePackagerBase, Protocol):
    """Focused test seam for a precomputed compile command."""

    def _make_compile_cmd(self, output_path: Path) -> list[str]: ...


CompilePackager = NativeCompilePackager | CommandCompilePackager


@dataclass(frozen=True, slots=True)
class CompilePhaseResult:
    attempted: bool
    succeeded: bool
    artifact_path: Path | None
    stdout: str
    filtered_stderr: str
    returncode: int


def _compile_command(
    packager: CompilePackager, output_path: Path
) -> tuple[list[str], Path]:
    if hasattr(packager, "_make_compile_cmd"):
        command_packager = cast(CommandCompilePackager, packager)
        return command_packager._make_compile_cmd(output_path), output_path

    native_packager = cast(NativeCompilePackager, packager)
    cmd, artifact_path = native_packager.compile()
    return cmd, Path(artifact_path)


def run_compile_phase(
    packager: CompilePackager,
    *,
    staging_dir: Path,
    compile_timeout: int,
    env_builder: Callable[
        [Mapping[str, str]], dict[str, str]
    ] = flashinfer_safetensors_env,
    runner: TextSubprocessRunner | None = None,
) -> CompilePhaseResult:
    """Compile a staged HIP/C++ solution and return subprocess diagnostics."""

    if not packager._is_cpp:
        return CompilePhaseResult(
            attempted=False,
            succeeded=False,
            artifact_path=None,
            stdout="",
            filtered_stderr="",
            returncode=0,
        )

    artifact_path = staging_dir / "benchmark_kernel.so"
    cmd, artifact_path = _compile_command(packager, artifact_path)
    (staging_dir / ".tmp").mkdir(exist_ok=True)
    base = sanitized_subprocess_env(os.environ, staging_dir=staging_dir)
    env = sanitized_subprocess_env(env_builder(base), staging_dir=staging_dir)
    if runner is None:
        proc = run_in_process_group_bounded(
            cmd, cwd=staging_dir, timeout=compile_timeout, env=env
        )
    else:
        proc = runner(
            cmd,
            cwd=staging_dir,
            capture_output=True,
            text=True,
            timeout=compile_timeout,
            env=env,
        )

    return CompilePhaseResult(
        attempted=True,
        succeeded=proc.returncode == 0,
        artifact_path=artifact_path,
        stdout=proc.stdout,
        filtered_stderr=filter_benign_rocm_stderr(proc.stderr),
        returncode=proc.returncode,
    )
