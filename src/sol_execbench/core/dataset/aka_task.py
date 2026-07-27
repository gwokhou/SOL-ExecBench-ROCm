# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Helpers used by AKA problem authoring and equivalence checks."""

from __future__ import annotations

import ast
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class AkaTask:
    """A resolved handle to an AKA task on disk."""

    aka_root: Path
    task_path: str
    config: dict[str, Any]

    @property
    def root(self) -> Path:
        """Return the task directory."""
        return self.aka_root / self.task_path

    @property
    def task_type(self) -> str:
        """Return the declared AKA task type."""
        return str(self.config.get("task_type") or "")

    @property
    def target_kernel_functions(self) -> tuple[str, ...]:
        """Return the kernel functions targeted by the task."""
        return tuple(
            str(s) for s in (self.config.get("target_kernel_functions") or [])
        )


def read_task(aka_root: str | Path, task_path: str) -> AkaTask:
    """Parse the ``config.yaml`` of an AKA task."""
    root = Path(aka_root).resolve()
    config_path = root / task_path / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"AKA task config not found: {config_path}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return AkaTask(root, task_path, config)


def functional_reference_path(task: AkaTask) -> Path:
    """Resolve the functional PyTorch reference for a torch2hip task."""
    candidate = _py_func_file_from_config(task)
    if candidate is None:
        func_dir = task.root / "pytorch_code_functional"
        files = sorted(func_dir.glob("*.py")) if func_dir.is_dir() else []
        if len(files) != 1:
            raise FileNotFoundError(
                f"could not resolve a unique functional reference for {task.task_path}",
            )
        candidate = files[0]
    return candidate if candidate.is_absolute() else (task.root / candidate)


def correctness_runner_path(task: AkaTask) -> Path:
    """Resolve the Python file named by a task's correctness command."""
    commands = task.config.get("correctness_command") or []
    for command in commands:
        try:
            tokens = shlex.split(str(command))
        except ValueError:
            continue
        candidates = [
            task.root / token
            for token in tokens
            if token.endswith(".py") and not token.startswith("-")
        ]
        files = [path for path in candidates if path.is_file()]
        if files:
            return files[-1]
    raise FileNotFoundError(
        f"could not resolve correctness runner for {task.task_path}",
    )


def _py_func_file_from_config(task: AkaTask) -> Path | None:
    for command in task.config.get("correctness_command") or []:
        try:
            tokens = shlex.split(str(command))
        except ValueError:
            continue
        for idx, token in enumerate(tokens):
            if token == "--py_func_file" and idx + 1 < len(tokens):
                return task.root / tokens[idx + 1]
            if token.startswith("--py_func_file="):
                return task.root / token.split("=", 1)[1]
    return None


def function_arg_names(text: str, function_name: str) -> list[str]:
    """Return the argument names of a top-level function (excluding 'self')."""
    module = ast.parse(text)
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            args = [a.arg for a in node.args.args]
            return [a for a in args if a != "self"]
    raise KeyError(f"function {function_name!r} not found at top level")


__all__ = [
    "AkaTask",
    "correctness_runner_path",
    "function_arg_names",
    "functional_reference_path",
    "read_task",
]
