# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Git source-state verification for publication-grade execution."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from sol_execbench.core.platform.runtime import resolve_tool_path
from sol_execbench.core.process.subprocesses import run_in_process_group_bounded

_GIT_TIMEOUT_SECONDS = 10.0
_MAX_GIT_OUTPUT_BYTES = 64 * 1024


@dataclass(frozen=True, slots=True)
class GitSourceState:
    """Observed Git revision and cleanliness for release-relevant paths."""

    revision: str
    tracked_dirty: bool
    untracked_paths: tuple[str, ...]

    @property
    def clean(self) -> bool:
        """Return whether no tracked or untracked release source differs."""
        return not self.tracked_dirty and not self.untracked_paths


def verify_git_source_state(
    repository_root: Path,
    *,
    expected_revision: str,
    paths: tuple[str, ...],
) -> GitSourceState:
    """Require exact committed source for a content-addressed release run."""
    if not paths:
        raise ValueError("release source path inventory must not be empty")
    root = repository_root.resolve()
    git = resolve_tool_path("git")
    if git is None:
        raise RuntimeError("Git is required to verify release source state")
    top_level = Path(_git_output(git, root, "rev-parse", "--show-toplevel")).resolve()
    if top_level != root:
        raise ValueError("release source root is not the Git repository root")
    revision = _git_output(git, root, "rev-parse", "HEAD")
    tracked = _run_git(git, root, "diff", "--quiet", "HEAD", "--", *paths)
    if tracked.returncode not in {0, 1}:
        raise RuntimeError("could not inspect tracked release source state")
    untracked_output = _git_output(
        git,
        root,
        "ls-files",
        "--others",
        "--exclude-standard",
        "--",
        *paths,
    )
    state = GitSourceState(
        revision=revision,
        tracked_dirty=tracked.returncode == 1,
        untracked_paths=tuple(line for line in untracked_output.splitlines() if line),
    )
    if state.revision != expected_revision:
        raise ValueError(
            "release source revision mismatch: "
            f"expected {expected_revision}, observed {state.revision}"
        )
    if not state.clean:
        raise ValueError("release source paths contain uncommitted changes")
    return state


def _git_output(git: Path, root: Path, *arguments: str) -> str:
    completed = _run_git(git, root, *arguments)
    if completed.returncode != 0:
        raise RuntimeError(f"Git source-state command failed: {' '.join(arguments)}")
    return completed.stdout.strip()


def _run_git(
    git: Path,
    root: Path,
    *arguments: str,
) -> subprocess.CompletedProcess[str]:
    try:
        return run_in_process_group_bounded(
            [str(git), "-C", str(root), *arguments],
            timeout=_GIT_TIMEOUT_SECONDS,
            max_capture_bytes=_MAX_GIT_OUTPUT_BYTES,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError("Git source-state command could not complete") from exc


__all__ = ["GitSourceState", "verify_git_source_state"]
