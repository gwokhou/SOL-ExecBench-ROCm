# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for eval-driver runtime helpers."""

from __future__ import annotations

import json
import math
import sys
import types
from io import StringIO

import pytest

from sol_execbench.core import Solution
from sol_execbench.core.bench.eval_runtime import (
    block_cpp_extension_load,
    emit_trace_jsonl,
    load_reference_function,
    load_staged_problem,
    load_user_function,
    measure_reference_latency,
    parse_entry_point,
    run_reward_hack_check,
    solution_uses_native_rocm,
)
from sol_execbench.core.bench.reward_hack import RewardHackDetected
from sol_execbench_type_helpers import make_trace, make_workload


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


def test_emit_trace_jsonl_writes_strict_trace_payload():
    stream = StringIO()
    trace = make_trace(
        definition="demo",
        solution="solution",
        workload=make_workload(uuid="workload", axes={}, inputs={}),
        evaluation=None,
    )

    emit_trace_jsonl(trace, stream)

    payload = json.loads(stream.getvalue())
    assert payload["definition"] == "demo"
    assert payload["workload"]["uuid"] == "workload"


def test_emit_trace_jsonl_rejects_non_finite_trace_values():
    stream = StringIO()

    class BadTrace:
        def model_dump(self, *, mode):
            assert mode == "json"
            return {"bad": math.nan}

    with pytest.raises(ValueError, match="Out of range float"):
        emit_trace_jsonl(BadTrace(), stream)


def test_run_reward_hack_check_returns_detected_message():
    def check():
        raise RewardHackDetected("patched timing")

    assert run_reward_hack_check(check) == "patched timing"


def test_run_reward_hack_check_suppresses_unexpected_errors_when_requested():
    def check():
        raise RuntimeError("diagnostic unavailable")

    assert run_reward_hack_check(check, suppress_errors=True) is None


def test_run_reward_hack_check_reraises_unexpected_errors_by_default():
    def check():
        raise RuntimeError("diagnostic unavailable")

    with pytest.raises(RuntimeError, match="diagnostic unavailable"):
        run_reward_hack_check(check)


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


def test_load_user_function_ignores_existing_simple_module_collision(tmp_path):
    (tmp_path / "kernel.py").write_text("def run():\n    return 'staged'\n")
    collision = types.ModuleType("kernel")
    collision.run = lambda: "collision"
    previous = sys.modules.get("kernel")
    sys.modules["kernel"] = collision
    try:
        fn = load_user_function(_solution(), tmp_path)
    finally:
        if previous is None:
            sys.modules.pop("kernel", None)
        else:
            sys.modules["kernel"] = previous

    assert fn() == "staged"


def test_load_user_function_ignores_existing_package_module_collision(tmp_path):
    package_dir = tmp_path / "pkg"
    package_dir.mkdir()
    (package_dir / "helper.py").write_text("VALUE = 'staged'\n")
    (package_dir / "kernel.py").write_text(
        "from .helper import VALUE\n\n"
        "def run():\n"
        "    return VALUE\n"
    )
    collision = types.ModuleType("pkg.kernel")
    collision.run = lambda: "collision"
    previous = sys.modules.get("pkg.kernel")
    sys.modules["pkg.kernel"] = collision
    try:
        fn = load_user_function(_solution(entry_point="pkg/kernel.py::run"), tmp_path)
    finally:
        if previous is None:
            sys.modules.pop("pkg.kernel", None)
        else:
            sys.modules["pkg.kernel"] = previous

    assert fn() == "staged"


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
