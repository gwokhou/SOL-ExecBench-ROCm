# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Compilation subprocess helpers for the SOL-ExecBench CLI."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from sol_execbench.core.bench.io import flashinfer_safetensors_env
from sol_execbench.core.bench.stderr import filter_benign_rocm_stderr
from sol_execbench.core.integrity import sha256_file
from sol_execbench.core.platform.runtime import resolve_rocm_tool
from sol_execbench.core.process.environment import sanitized_subprocess_env
from sol_execbench.core.process.subprocesses import (
    TextSubprocessRunner,
    run_in_process_group_bounded,
)


class CompilePackager(Protocol):
    """Staged-package behavior needed by the compile phase."""

    @property
    def _is_cpp(self) -> bool: ...

    def compile(self) -> tuple[list[str], str]:
        """Return the compilation command and resulting artifact path."""
        ...


@dataclass(frozen=True, slots=True)
class CompilePhaseResult:
    """Compilation outcome and optional terminal CLI result."""

    attempted: bool
    succeeded: bool
    artifact_path: Path | None
    stdout: str
    filtered_stderr: str
    returncode: int
    command: tuple[str, ...] = ()
    compiler_path: str | None = None
    compiler_sha256: str | None = None
    compiler_version: str | None = None


def run_compile_phase(
    packager: CompilePackager,
    *,
    staging_dir: Path,
    compile_timeout: int,
    env_builder: Callable[
        [Mapping[str, str]],
        dict[str, str],
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
            command=(),
        )

    cmd, artifact_path_text = packager.compile()
    artifact_path = Path(artifact_path_text)
    (staging_dir / ".tmp").mkdir(exist_ok=True)
    base = sanitized_subprocess_env(os.environ, staging_dir=staging_dir)
    env = sanitized_subprocess_env(env_builder(base), staging_dir=staging_dir)
    if runner is None:
        proc = run_in_process_group_bounded(
            cmd,
            cwd=staging_dir,
            timeout=compile_timeout,
            env=env,
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

    compiler = (
        _compiler_provenance()
        if runner is None and proc.returncode == 0
        else (None, None, None)
    )
    return CompilePhaseResult(
        attempted=True,
        succeeded=proc.returncode == 0,
        artifact_path=artifact_path,
        stdout=proc.stdout,
        filtered_stderr=filter_benign_rocm_stderr(proc.stderr),
        returncode=proc.returncode,
        command=tuple(cmd),
        compiler_path=compiler[0],
        compiler_sha256=compiler[1],
        compiler_version=compiler[2],
    )


def _compiler_provenance() -> tuple[str | None, str | None, str | None]:
    compiler = resolve_rocm_tool("hipcc")
    if compiler is None:
        return None, None, None
    completed = run_in_process_group_bounded(
        [str(compiler), "--version"],
        timeout=30.0,
    )
    if completed.returncode != 0:
        return str(compiler), sha256_file(compiler), None
    version = next(
        (
            line.strip()
            for line in completed.stdout.splitlines()
            if line.strip()
        ),
        None,
    )
    return str(compiler), sha256_file(compiler), version
