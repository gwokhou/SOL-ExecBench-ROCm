# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for eval-driver runtime helpers."""

from __future__ import annotations

import json

import pytest

from sol_execbench.core import Solution
from sol_execbench.core.bench.eval_runtime import (
    block_cpp_extension_load,
    load_reference_function,
    load_staged_problem,
    load_user_function,
    measure_reference_latency,
    parse_entry_point,
    solution_uses_native_rocm,
)


def _solution(entry_point: str = "kernel.py::run", languages: list[str] | None = None):
    source_path = entry_point.split("::", 1)[0]
    return Solution(
        name="demo",
        definition="demo",
        author="test",
        spec={
            "languages": languages or ["pytorch"],
            "target_hardware": ["LOCAL"],
            "entry_point": entry_point,
            "destination_passing_style": False,
        },
        sources=[{"path": source_path, "content": "def run(x):\n    return x\n"}],
    )


def test_load_staged_problem_reads_definition_and_workloads(tmp_path):
    (tmp_path / "definition.json").write_text(json.dumps({"name": "demo"}))
    (tmp_path / "workload.jsonl").write_text(
        json.dumps({"uuid": "a"}) + "\n\n" + json.dumps({"uuid": "b"}) + "\n"
    )

    definition, workloads = load_staged_problem(tmp_path)

    assert definition == {"name": "demo"}
    assert workloads == [{"uuid": "a"}, {"uuid": "b"}]


def test_load_staged_problem_rejects_missing_definition(tmp_path):
    with pytest.raises(RuntimeError, match="definition.json not found"):
        load_staged_problem(tmp_path)


def test_parse_entry_point_defaults_to_run():
    assert parse_entry_point("kernel.py::entry") == ("kernel.py", "entry")
    assert parse_entry_point("kernel") == ("kernel", "run")


def test_load_reference_function_writes_real_module_file(tmp_path):
    module, fn = load_reference_function(tmp_path, "def run(x):\n    return x + 1\n")

    assert (tmp_path / "_reference.py").exists()
    assert fn(2) == 3
    assert vars(module)["run"] is fn


def test_load_reference_function_requires_run(tmp_path):
    with pytest.raises(RuntimeError, match="does not define"):
        load_reference_function(tmp_path, "VALUE = 1\n")


def test_measure_reference_latency_returns_float_latency():
    def fake_time_fn(*args, **kwargs):
        return 1.25

    result = measure_reference_latency(
        lambda x: x,
        [1],
        "cpu",
        warmup=1,
        rep=2,
        time_fn=fake_time_fn,
    )

    assert result.latency_ms == 1.25
    assert result.failure is None


def test_measure_reference_latency_returns_failure_message_on_exception():
    def fake_time_fn(*args, **kwargs):
        raise RuntimeError("boom")

    result = measure_reference_latency(
        lambda x: x,
        [1],
        "cpu",
        warmup=1,
        rep=2,
        time_fn=fake_time_fn,
    )

    assert result.latency_ms == 0.0
    assert result.failure == "Reference timing failed: boom"


def test_measure_reference_latency_rejects_non_numeric_latency():
    def fake_time_fn(*args, **kwargs):
        return "fast"

    result = measure_reference_latency(
        lambda x: x,
        [1],
        "cpu",
        warmup=1,
        rep=2,
        time_fn=fake_time_fn,
    )

    assert result.latency_ms == 0.0
    assert result.failure == "Reference timing returned non-numeric result: str"


def test_load_user_function_imports_python_solution(tmp_path):
    (tmp_path / "kernel.py").write_text("def run(x):\n    return x + 2\n")

    fn = load_user_function(_solution(), tmp_path)

    assert fn(3) == 5


def test_solution_uses_native_rocm_and_missing_so_rejected(tmp_path):
    solution = _solution(entry_point="kernel.hip::run", languages=["hip_cpp"])

    assert solution_uses_native_rocm(solution) is True
    with pytest.raises(RuntimeError, match="benchmark_kernel.so not found"):
        load_user_function(solution, tmp_path)


def test_block_cpp_extension_load_rejects_dynamic_compiles():
    import torch.utils.cpp_extension as cpp_ext

    original_load = cpp_ext.load
    original_load_inline = cpp_ext.load_inline
    try:
        block_cpp_extension_load()

        with pytest.raises(RuntimeError, match="not permitted"):
            cpp_ext.load("demo", [])
        with pytest.raises(RuntimeError, match="not permitted"):
            cpp_ext.load_inline(name="demo", cpp_sources="")
    finally:
        cpp_ext.load = original_load
        cpp_ext.load_inline = original_load_inline
