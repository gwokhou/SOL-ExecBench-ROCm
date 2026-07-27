# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the AKA task parser helpers (no GPU, no AKA clone required)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from sol_execbench.core.dataset.aka_task import (
    AKATask,
    correctness_runner_path,
    function_arg_names,
    functional_reference_path,
    read_task,
)

FIXTURE = (
    Path(__file__).resolve().parents[2] / "fixtures" / "aka" / "sample_func.py"
)


def test_function_arg_names_returns_inputs_in_order():
    text = FIXTURE.read_text()

    assert function_arg_names(text, "module_fn") == ["a", "b"]
    assert function_arg_names(text, "get_inputs") == []


def test_function_arg_names_rejects_missing_function():
    with pytest.raises(KeyError, match="does_not_exist"):
        function_arg_names(FIXTURE.read_text(), "does_not_exist")


def _task(tmp_path: Path, config: dict[str, object]) -> AKATask:
    root = tmp_path / "tasks/torch2hip/example"
    root.mkdir(parents=True)
    (root / "config.yaml").write_text(yaml.safe_dump(config))
    return read_task(tmp_path, "tasks/torch2hip/example")


def test_read_task_exposes_closed_properties_and_missing_config(
    tmp_path,
) -> None:
    task = _task(
        tmp_path,
        {
            "task_type": "torch2hip",
            "target_kernel_functions": ["run", "backward"],
        },
    )

    assert task.root == tmp_path / "tasks/torch2hip/example"
    assert task.task_type == "torch2hip"
    assert task.target_kernel_functions == ("run", "backward")

    with pytest.raises(FileNotFoundError, match="AKA task config not found"):
        read_task(tmp_path, "tasks/missing")


@pytest.mark.parametrize(
    "argument",
    [
        "--py_func_file reference.py",
        "--py_func_file=reference.py",
    ],
)
def test_functional_reference_resolves_command_argument(
    tmp_path,
    argument: str,
) -> None:
    task = _task(
        tmp_path,
        {"correctness_command": [f"python check.py {argument}"]},
    )
    reference = task.root / "reference.py"
    reference.write_text("def run():\n    return 1\n")

    assert functional_reference_path(task) == reference


def test_functional_reference_fallback_requires_exactly_one_file(
    tmp_path,
) -> None:
    task = _task(tmp_path, {})
    directory = task.root / "pytorch_code_functional"
    directory.mkdir()

    with pytest.raises(FileNotFoundError, match="unique functional reference"):
        functional_reference_path(task)

    first = directory / "first.py"
    first.write_text("VALUE = 1\n")
    assert functional_reference_path(task) == first

    (directory / "second.py").write_text("VALUE = 2\n")
    with pytest.raises(FileNotFoundError, match="unique functional reference"):
        functional_reference_path(task)


def test_correctness_runner_ignores_invalid_commands_and_uses_last_python_file(
    tmp_path,
) -> None:
    task = _task(
        tmp_path,
        {
            "correctness_command": [
                "unterminated '",
                "python helper.py runner.py --flag",
            ],
        },
    )
    (task.root / "helper.py").write_text("")
    runner = task.root / "runner.py"
    runner.write_text("")

    assert correctness_runner_path(task) == runner

    runner.unlink()
    (task.root / "helper.py").unlink()
    with pytest.raises(FileNotFoundError, match="correctness runner"):
        correctness_runner_path(task)
