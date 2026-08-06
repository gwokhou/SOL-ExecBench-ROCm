# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Capability-matrix tests for the v6 Extended native IR."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml
from torch.nn import functional

from solar.analysis.graph_analyzer import IRGraphAnalyzer
from solar.graph.contracts import ExtractionKind
from solar.graph.extraction import extract_operator_graph
from solar.ir.contracts import IRKind
from solar.ir.conversion import convert_operator_graph
from solar.ir.extended_einsum.backend import backend as extended_backend
from solar.ir.extended_einsum.native_registry import (
    NATIVE_OP_REGISTRY,
    canonical_native_target,
)
from solar.ir.registry import ir_backend
from solar.schema_versions import EXTENDED_EINSUM_IR_SCHEMA_VERSION
from solar.verification.executor import IRGraphExecutor
from solar.verification.registry import verification_backend

_EXTENDED_VERIFIER = verification_backend(IRKind.EXTENDED_EINSUM)


def _cases() -> list[tuple[str, Callable[..., Any], tuple[torch.Tensor, ...]]]:
    matrix = torch.arange(12.0).reshape(3, 4)
    image = torch.arange(32.0).reshape(2, 1, 4, 4)
    grid = torch.zeros(2, 2, 2, 2)
    indices = torch.tensor([[0, 2], [1, 0], [3, 2]])
    logits = torch.arange(15.0).reshape(3, 5)
    embedding_indices = torch.tensor([0, 1, 2, 1])
    embedding_weight = torch.arange(18.0).reshape(6, 3)
    offsets = torch.tensor([0, 2])
    convolution_input = torch.arange(18.0).reshape(1, 2, 3, 3)
    convolution_weight = torch.arange(24.0).reshape(2, 3, 2, 2)
    return [
        (
            "add",
            lambda x, y: torch.add(x, y, alpha=2),
            (matrix, matrix.clone()),
        ),
        ("bfloat16", lambda x: x.bfloat16(), (matrix,)),
        ("float", lambda x: x.float(), (matrix.to(torch.bfloat16),)),
        ("half", lambda x: x.half(), (matrix,)),
        ("int", lambda x: x.int(), (matrix,)),
        ("long", lambda x: x.long(), (matrix,)),
        ("ones_like", torch.ones_like, (matrix,)),
        ("eq", lambda x: x == 2, (matrix,)),
        ("reshape", lambda x: x.reshape(2, 2, 3), (matrix,)),
        (
            "amax",
            lambda x: torch.amax(x, dim=-1, keepdim=True),
            (matrix,),
        ),
        (
            "clamp",
            lambda x: torch.clamp(x, -2.0, 5.0),
            (matrix - 4.0,),
        ),
        (
            "matmul",
            torch.matmul,
            (matrix, torch.arange(20.0).reshape(4, 5)),
        ),
        (
            "rms_norm",
            lambda x, w: functional.rms_norm(x, (4,), w, 1e-5),
            (matrix, torch.ones(4)),
        ),
        ("silu", functional.silu, (matrix,)),
        (
            "softplus",
            lambda x: functional.softplus(x, beta=1.5, threshold=12.0),
            (matrix,),
        ),
        ("softmax", lambda x: functional.softmax(x, dim=-1), (matrix,)),
        (
            "sort",
            lambda x: torch.sort(x, dim=-1, descending=True, stable=True),
            (matrix,),
        ),
        (
            "argsort",
            lambda x: torch.argsort(x, dim=-1, descending=True, stable=True),
            (matrix,),
        ),
        ("topk", lambda x: torch.topk(x, 2, dim=-1), (matrix,)),
        (
            "var",
            lambda x: torch.var(x, dim=-1, correction=0, keepdim=True),
            (matrix,),
        ),
        (
            "std",
            lambda x: torch.std(x, dim=-1, correction=0, keepdim=True),
            (matrix,),
        ),
        ("all", lambda x: torch.all(x > 2, dim=-1, keepdim=True), (matrix,)),
        ("any", lambda x: torch.any(x > 2, dim=-1, keepdim=True), (matrix,)),
        (
            "max_pool2d",
            lambda x: functional.max_pool2d(
                x, 2, stride=1, padding=1, return_indices=True
            ),
            (image,),
        ),
        (
            "avg_pool2d",
            lambda x: functional.avg_pool2d(
                x, 2, stride=1, padding=1, count_include_pad=False
            ),
            (image,),
        ),
        (
            "adaptive_avg_pool2d",
            lambda x: functional.adaptive_avg_pool2d(x, (2, 3)),
            (image,),
        ),
        (
            "interpolate",
            lambda x: functional.interpolate(
                x, size=(5, 5), mode="bilinear", align_corners=False
            ),
            (image,),
        ),
        (
            "grid_sample",
            lambda x, g: functional.grid_sample(
                x,
                g,
                mode="bilinear",
                padding_mode="border",
                align_corners=False,
            ),
            (image, grid),
        ),
        ("gather", lambda x, i: torch.gather(x, 1, i), (matrix, indices)),
        (
            "scatter_add",
            lambda x, i, s: torch.scatter_add(x, 1, i, s),
            (
                torch.zeros_like(matrix),
                indices,
                torch.ones_like(indices).float(),
            ),
        ),
        (
            "index_add",
            lambda x, i, s: torch.index_add(x, 0, i, s, alpha=0.5),
            (
                torch.zeros(3, 4),
                torch.tensor([0, 2]),
                torch.ones(2, 4),
            ),
        ),
        (
            "index_select",
            lambda x, i: torch.index_select(x, -1, i),
            (matrix, torch.tensor([3, 1])),
        ),
        (
            "repeat_interleave",
            lambda x: torch.repeat_interleave(x, 2, dim=-1),
            (matrix,),
        ),
        (
            "diagonal",
            lambda x: torch.diagonal(x, offset=1, dim1=-2, dim2=-1),
            (matrix,),
        ),
        ("flip", lambda x: torch.flip(x, dims=(-1,)), (matrix,)),
        (
            "roll",
            lambda x: torch.roll(x, shifts=(1, -1), dims=(0, 1)),
            (matrix,),
        ),
        (
            "pad",
            lambda x: functional.pad(x, (1, 2), mode="constant", value=-1),
            (matrix,),
        ),
        (
            "einsum",
            lambda x, y: torch.einsum("mk,kn->mn", x, y),
            (matrix, torch.arange(20.0).reshape(4, 5)),
        ),
        (
            "scaled_dot_product_attention",
            lambda q, k, v: functional.scaled_dot_product_attention(
                q, k, v, dropout_p=0.0, is_causal=True, scale=0.5
            ),
            (
                torch.randn(1, 2, 3, 4),
                torch.randn(1, 2, 3, 4),
                torch.randn(1, 2, 3, 5),
            ),
        ),
        (
            "cross_entropy",
            lambda x, y: functional.cross_entropy(
                x, y, reduction="sum", label_smoothing=0.1
            ),
            (logits, torch.tensor([0, 2, 4])),
        ),
        (
            "embedding_bag",
            lambda i, w, o: functional.embedding_bag(
                i, w, o, mode="mean", include_last_offset=False
            ),
            (embedding_indices, embedding_weight, offsets),
        ),
        (
            "fft",
            lambda x: torch.fft.fft(x, n=8, dim=-1, norm="ortho"),
            (matrix,),
        ),
        (
            "vector_norm",
            lambda x: torch.linalg.vector_norm(x, ord=2, dim=-1, keepdim=True),
            (matrix,),
        ),
        ("nonzero", lambda x: torch.nonzero(x, as_tuple=False), (matrix,)),
        (
            "conv_transpose2d",
            lambda x, w: functional.conv_transpose2d(
                x, w, stride=2, padding=1, output_padding=1
            ),
            (convolution_input, convolution_weight),
        ),
    ]


def _gradient_scalar(value: Any) -> torch.Tensor | None:
    leaves: list[torch.Tensor] = []
    values = value if isinstance(value, (tuple, list)) else (value,)
    for item in values:
        if (
            isinstance(item, torch.Tensor)
            and item.requires_grad
            and (item.is_floating_point() or item.is_complex())
        ):
            leaves.append(item)
    if not leaves:
        return None
    scalars = [
        item.real.sum() + item.imag.sum() if item.is_complex() else item.sum()
        for item in leaves
    ]
    return scalars[0] + sum(scalars[1:], start=torch.zeros_like(scalars[0]))


def _gradients(
    callable_: Callable[..., Any],
    inputs: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor | None, ...]:
    watched = tuple(
        item for item in inputs if item.is_floating_point() or item.is_complex()
    )
    scalar = _gradient_scalar(callable_(*inputs))
    if scalar is None:
        return tuple(None for _ in watched)
    return torch.autograd.grad(scalar, watched, allow_unused=True)


def _assert_gradient_parity(
    reference: Callable[..., Any],
    executor: IRGraphExecutor,
    inputs: tuple[torch.Tensor, ...],
) -> None:
    reference_inputs = tuple(
        item.detach()
        .clone()
        .requires_grad_(item.is_floating_point() or item.is_complex())
        for item in inputs
    )
    executor_inputs = tuple(
        item.detach()
        .clone()
        .requires_grad_(item.is_floating_point() or item.is_complex())
        for item in inputs
    )
    expected = _gradients(reference, reference_inputs)
    actual = _gradients(executor, executor_inputs)
    assert len(actual) == len(expected)
    for observed, wanted in zip(actual, expected, strict=True):
        assert (observed is None) == (wanted is None)
        if observed is not None:
            torch.testing.assert_close(observed, wanted, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize(("name", "reference", "inputs"), _cases())
def test_native_v6_cpu_capability_matrix(
    tmp_path: Path,
    name: str,
    reference: Callable[..., Any],
    inputs: tuple[torch.Tensor, ...],
) -> None:
    output = tmp_path / name
    operator = extract_operator_graph(
        reference,
        inputs,
        device="cpu",
        output_dir=output,
        name=name,
        extraction_kind=ExtractionKind.TORCHVIEW,
    )
    converted = convert_operator_graph(operator, output_dir=output)
    graph = yaml.safe_load(converted.path.read_text())
    actual = IRGraphExecutor(graph, _EXTENDED_VERIFIER)(*inputs)
    expected = reference(*inputs)

    assert graph["schema_version"] == EXTENDED_EINSUM_IR_SCHEMA_VERSION
    torch.testing.assert_close(actual, expected)
    serialized = converted.path.read_text()
    assert "kind: aten" not in serialized
    assert "exact_target:" not in serialized
    assert "overload:" not in serialized
    analysis = IRGraphAnalyzer(
        validator=extended_backend.validate
    ).analyze_graph(
        converted.path,
        output / "analysis",
        copy_graph=False,
        strict=True,
    )
    assert analysis is not None
    _assert_gradient_parity(
        reference, IRGraphExecutor(graph, _EXTENDED_VERIFIER), inputs
    )


def test_native_registry_and_executor_are_independent_from_aten() -> None:
    from solar.verification import extended

    required = {name for name, _, _ in _cases()} - {"einsum"}
    assert required <= set(NATIVE_OP_REGISTRY)
    assert _EXTENDED_VERIFIER.execute is extended.execute_extended_einsum_layer
    source = inspect.getsource(extended)
    assert "execute_aten_layer" not in source
    assert "torch.ops.aten" not in source


@pytest.mark.parametrize(
    ("public_name", "canonical"),
    [
        ("__and__", "bitwise_and"),
        ("__eq__", "eq"),
        ("__ge__", "ge"),
        ("__gt__", "gt"),
        ("__invert__", "bitwise_not"),
        ("__le__", "le"),
        ("__lt__", "lt"),
        ("__matmul__", "matmul"),
        ("__ne__", "ne"),
        ("__neg__", "neg"),
    ],
)
def test_supported_python_operators_keep_public_aliases(
    public_name: str,
    canonical: str,
) -> None:
    assert canonical_native_target(public_name) == canonical


@pytest.mark.parametrize(("name", "reference", "inputs"), _cases())
def test_aten_v6_cpu_capability_matrix(
    tmp_path: Path,
    name: str,
    reference: Callable[..., Any],
    inputs: tuple[torch.Tensor, ...],
) -> None:
    output = tmp_path / f"aten-{name}"
    operator = extract_operator_graph(
        reference,
        inputs,
        device="cpu",
        output_dir=output,
        name=name,
        extraction_kind=ExtractionKind.MAKE_FX_REFERENCE,
    )
    converted = convert_operator_graph(
        operator,
        output_dir=output,
        ir_kind=IRKind.ATEN,
    )
    graph = yaml.safe_load(converted.path.read_text())
    aten_backend = ir_backend(IRKind.ATEN)
    executor = IRGraphExecutor(graph, verification_backend(IRKind.ATEN))

    torch.testing.assert_close(executor(*inputs), reference(*inputs))
    analysis = IRGraphAnalyzer(validator=aten_backend.validate).analyze_graph(
        converted.path,
        output / "analysis",
        copy_graph=False,
        strict=True,
    )
    assert analysis is not None
    _assert_gradient_parity(reference, executor, inputs)


def test_nonzero_uses_one_bounded_runtime_symbol(tmp_path: Path) -> None:
    def reference(value):
        return torch.nonzero(value, as_tuple=True)

    traced = torch.tensor([[0.0, 1.0], [2.0, 0.0]])
    operator = extract_operator_graph(
        reference,
        (traced,),
        device="cpu",
        output_dir=tmp_path,
        name="nonzero-tuple",
        extraction_kind=ExtractionKind.TORCHVIEW,
    )
    converted = convert_operator_graph(operator, output_dir=tmp_path)
    graph = yaml.safe_load(converted.path.read_text())
    nonzero = next(
        layer
        for layer in graph["layers"].values()
        if (layer.get("semantic_op") or {}).get("target") == "nonzero"
    )
    descriptors = [shape[0] for shape in nonzero["tensor_shapes"]["outputs"]]
    assert {item["symbol"] for item in descriptors} == {"nnz0"}
    assert {item["upper"] for item in descriptors} == {4}

    executor = IRGraphExecutor(graph, _EXTENDED_VERIFIER)
    for value in (torch.zeros(2, 2), torch.ones(2, 2)):
        actual = executor(value)
        expected = reference(value)
        torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    ("name", "reference", "inputs", "mutation_index"),
    [
        (
            "add-out",
            lambda left, right, out: torch.add(left, right, alpha=2, out=out),
            (torch.ones(4), torch.full((4,), 2.0), torch.empty(4)),
            2,
        ),
        (
            "add-inplace",
            lambda left, right: left.add_(right, alpha=2),
            (torch.ones(4), torch.full((4,), 2.0)),
            0,
        ),
    ],
)
def test_native_effects_preserve_out_and_inplace_aliases(
    tmp_path: Path,
    name: str,
    reference: Callable[..., torch.Tensor],
    inputs: tuple[torch.Tensor, ...],
    mutation_index: int,
) -> None:
    operator = extract_operator_graph(
        reference,
        inputs,
        device="cpu",
        output_dir=tmp_path / name,
        name=name,
        extraction_kind=ExtractionKind.TORCHVIEW,
    )
    converted = convert_operator_graph(operator, output_dir=tmp_path / name)
    graph = yaml.safe_load(converted.path.read_text())
    operation = next(
        layer["semantic_op"]
        for layer in graph["layers"].values()
        if (layer.get("semantic_op") or {}).get("kind") == "operation"
    )
    assert operation["effects"]["mutates"] == [mutation_index]
    assert operation["effects"]["aliases"] == [
        {"output": 0, "input": mutation_index}
    ]

    reference_inputs = tuple(item.clone() for item in inputs)
    executor_inputs = tuple(item.clone() for item in inputs)
    expected = reference(*reference_inputs)
    actual = IRGraphExecutor(graph, _EXTENDED_VERIFIER)(*executor_inputs)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        executor_inputs[mutation_index], reference_inputs[mutation_index]
    )
    assert actual is executor_inputs[mutation_index]
