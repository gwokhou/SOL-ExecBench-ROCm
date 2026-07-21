# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Helpers for reading AMD AgentKernelArena (AKA) tasks.

Used by the problem-authoring workflow and by ``scripts/aka_equivalence_check.py``
to (a) lift the per-task PyTorch reference (``module_fn`` for torch2hip /
torch2flydsl; the test-file torch oracle for instruction2triton) and (b)
materialize the task's ``get_inputs()`` cases so an authored reference can be
cross-checked against AKA's own correctness oracle.
"""

from __future__ import annotations

import ast
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml


@dataclass(frozen=True)
class AkaTask:
    """A resolved handle to an AKA task on disk."""

    aka_root: Path
    task_path: str
    config: dict[str, Any]

    @property
    def root(self) -> Path:
        return self.aka_root / self.task_path

    @property
    def task_type(self) -> str:
        return str(self.config.get("task_type") or "")

    @property
    def target_kernel_functions(self) -> tuple[str, ...]:
        return tuple(str(s) for s in (self.config.get("target_kernel_functions") or []))


def read_task(aka_root: str | Path, task_path: str) -> AkaTask:
    """Parse the ``config.yaml`` of an AKA task."""
    root = Path(aka_root).resolve()
    config_path = root / task_path / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"AKA task config not found: {config_path}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return AkaTask(root, task_path, config)


def functional_reference_text(task: AkaTask) -> str:
    """Return the PyTorch functional-reference source for a torch2* task.

    Resolves the file from the ``correctness_command``'s ``--py_func_file``
    argument when present, falling back to the single file under
    ``pytorch_code_functional/``.
    """
    candidate = _py_func_file_from_config(task)
    if candidate is None:
        func_dir = task.root / "pytorch_code_functional"
        files = sorted(func_dir.glob("*.py")) if func_dir.is_dir() else []
        if len(files) != 1:
            raise FileNotFoundError(
                f"could not resolve a unique functional reference for {task.task_path}"
            )
        candidate = files[0]
    path = candidate if candidate.is_absolute() else (task.root / candidate)
    return path.read_text(encoding="utf-8")


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


def extract_function_source(text: str, function_name: str) -> str:
    """Return the source segment of a top-level ``def <function_name>``."""
    module = ast.parse(text)
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            return ast.get_source_segment(text, node) or ""
    raise KeyError(f"function {function_name!r} not found at top level")


def function_arg_names(text: str, function_name: str) -> list[str]:
    """Return the argument names of a top-level function (excluding 'self')."""
    module = ast.parse(text)
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            args = [a.arg for a in node.args.args]
            return [a for a in args if a != "self"]
    raise KeyError(f"function {function_name!r} not found at top level")


def materialize_get_inputs(task: AkaTask) -> list[list[Any]]:
    """Execute the functional file and collect ``get_inputs()`` yielded cases.

    Returns a list of input-lists (each a list of tensors/scalars). The file is
    executed in a fresh namespace with ``torch`` available; only the yielded
    cases are captured (values are not retained beyond this call).
    """
    import torch  # noqa: PLC0415 — deferred so non-torch envs can import the module

    text = functional_reference_text(task)
    namespace: dict[str, Any] = {"torch": torch}
    exec(compile(text, str(task.root), "exec"), namespace)  # noqa: S102 — trusted AKA source
    generator = namespace.get("get_inputs")
    if generator is None:
        raise KeyError(f"get_inputs not defined for {task.task_path}")
    cases = generator()
    if callable(cases):
        cases = cases()
    return [list(case) for case in cases]


def iter_suite_tasks(
    aka_root: str | Path, suites: Iterable[str]
) -> list[tuple[str, str]]:
    """Yield ``(suite, task_path)`` for every ``config.yaml`` under the suites."""
    root = Path(aka_root).resolve()
    out: list[tuple[str, str]] = []
    for suite in suites:
        base = root / "tasks" / suite
        if not base.is_dir():
            continue
        for config_path in sorted(base.rglob("config.yaml")):
            task_path = config_path.parent.relative_to(root).as_posix()
            out.append((suite, task_path))
    return out


__all__ = [
    "AkaTask",
    "extract_function_source",
    "function_arg_names",
    "functional_reference_text",
    "iter_suite_tasks",
    "materialize_get_inputs",
    "read_task",
]
