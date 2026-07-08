# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Reference-code validation helpers for workload definitions."""

from __future__ import annotations

import ast
from typing import Any


def validate_reference_code(definition: Any) -> Any:
    """Validate that reference contains valid Python code with a top-level run function."""
    try:
        mod = ast.parse(definition.reference, mode="exec")
    except SyntaxError as e:
        raise ValueError(f"Reference must be valid Python code: {e}") from e

    has_run_func = any(
        isinstance(node, ast.FunctionDef) and node.name == "run" for node in mod.body
    )
    if not has_run_func:
        raise ValueError("Reference must define a top-level function named 'run'")
    return definition


def validate_reference_inputs_match(definition: Any) -> Any:
    """Validate that run() parameter names match the definition inputs in order."""
    try:
        tree = ast.parse(definition.reference, mode="exec")
    except SyntaxError:
        return definition

    run_func = next(
        (n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run"),
        None,
    )
    if run_func is None:
        return definition

    args = run_func.args
    param_names: list[str] = [a.arg for a in (args.posonlyargs or [])] + [
        a.arg for a in args.args
    ]
    input_names = list(definition.inputs.keys())

    if len(param_names) != len(input_names):
        raise ValueError(
            f"run() has {len(param_names)} parameter(s) {param_names} but "
            f"definition 'inputs' has {len(input_names)} "
            f"entr{'y' if len(input_names) == 1 else 'ies'} "
            f"{input_names}. They must match exactly."
        )

    mismatched = [(p, i) for p, i in zip(param_names, input_names) if p != i]
    if mismatched:
        raise ValueError(
            f"run() parameter names don't match definition 'inputs' keys. "
            f"Mismatches (run_param -> input_key): {mismatched}. "
            f"run() params: {param_names}, inputs: {input_names}."
        )

    return definition


def verify_custom_inputs_entrypoint(definition: Any) -> Any:
    """Verify that the custom inputs entrypoint is a valid top-level reference function."""
    if definition.custom_inputs_entrypoint is None:
        return definition

    if not definition.custom_inputs_entrypoint.isidentifier():
        raise ValueError(
            f"custom_inputs_entrypoint must be a valid Python identifier, "
            f"got: {definition.custom_inputs_entrypoint!r}"
        )

    try:
        tree = ast.parse(definition.reference, mode="exec")
    except SyntaxError:
        return definition

    has_entrypoint = any(
        isinstance(node, ast.FunctionDef)
        and node.name == definition.custom_inputs_entrypoint
        for node in tree.body
    )
    if not has_entrypoint:
        raise ValueError(
            f"custom_inputs_entrypoint '{definition.custom_inputs_entrypoint}' "
            f"is not defined as a top-level function in the reference code"
        )

    return definition
