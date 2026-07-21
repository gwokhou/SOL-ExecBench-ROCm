# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the AKA task parser helpers (no GPU, no AKA clone required)."""

from __future__ import annotations

from pathlib import Path

from sol_execbench.core.dataset.aka_task import (
    extract_function_source,
    function_arg_names,
)

FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "aka" / "sample_func.py"


def test_extract_module_fn_source_lifts_the_function_body():
    text = FIXTURE.read_text()

    source = extract_function_source(text, "module_fn")

    assert source.startswith("def module_fn(a: torch.Tensor, b: torch.Tensor)")
    assert "torch.matmul(a, b + b)" in source


def test_function_arg_names_returns_inputs_in_order():
    text = FIXTURE.read_text()

    assert function_arg_names(text, "module_fn") == ["a", "b"]
    assert function_arg_names(text, "get_inputs") == []


def test_extract_function_source_rejects_missing_function():
    import pytest

    with pytest.raises(KeyError):
        extract_function_source(FIXTURE.read_text(), "does_not_exist")
