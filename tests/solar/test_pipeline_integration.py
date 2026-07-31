from __future__ import annotations

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
from solar.ir.registry import ir_backend
from solar.verification.registry import verification_backend
from solar.verification.verify import IRGraphExecutor


def _add_relu(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return torch.relu(left + right)


def _matmul(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return torch.matmul(left, right)


def _sum_keepdim(value: torch.Tensor) -> torch.Tensor:
    return torch.sum(value, dim=1, keepdim=True)


def _view_transpose(value: torch.Tensor) -> torch.Tensor:
    return value.reshape(3, 2).transpose(0, 1)


def _cat(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return torch.cat((left, right), dim=1)


def _stack(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return torch.stack((left, right), dim=0)


def _vstack(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return torch.vstack((left, right))


def _cumsum(value: torch.Tensor) -> torch.Tensor:
    return torch.cumsum(value, dim=1)


def _masked_fill(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return value.masked_fill(mask, -1.0)


def _masked_fill_negative_inf(
    value: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    return value.masked_fill(mask, float("-inf"))


def _where(
    mask: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
) -> torch.Tensor:
    return torch.where(mask, left, right)


def _conv2d(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return functional.conv2d(value, weight, padding=1)


def _embedding(indices: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return functional.embedding(indices, weight)


def _sdpa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    return functional.scaled_dot_product_attention(query, key, value)


def _linear(
    value: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    return functional.linear(value, weight, bias)


def _layer_norm(
    value: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    return functional.layer_norm(value, (3,), weight, bias, 1e-5)


def _max_values(value: torch.Tensor) -> torch.Tensor:
    return torch.max(value, dim=1).values


def _split_recombine(value: torch.Tensor) -> torch.Tensor:
    first, second, third = torch.split(value, 2, dim=1)
    return first + second + third


def _split_outputs(
    value: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    return torch.split(value, 2, dim=1)


def _topk_outputs(
    value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.topk(value, 3, dim=1)


def _scalar_tensor_chain(
    value: torch.Tensor,
    target: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    reduced = functional.kl_div(value / scale, target, reduction="sum")
    return reduced * (scale * scale) / value.shape[0]


def _batch_norm_with_tensor_kwargs(
    value: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
) -> torch.Tensor:
    return functional.batch_norm(
        value,
        running_mean,
        running_var,
        weight=weight,
        bias=bias,
        training=False,
    )


def _cases() -> list[tuple[str, Callable[..., torch.Tensor], tuple[Any, ...]]]:
    return [
        ("add_relu", _add_relu, (torch.ones(2, 3), torch.full((2, 3), 2.0))),
        (
            "matmul",
            _matmul,
            (torch.arange(6.0).reshape(2, 3), torch.arange(12.0).reshape(3, 4)),
        ),
        ("sum_keepdim", _sum_keepdim, (torch.arange(6.0).reshape(2, 3),)),
        ("view_transpose", _view_transpose, (torch.arange(6.0),)),
        ("cat", _cat, (torch.ones(2, 2), torch.zeros(2, 1))),
        ("stack", _stack, (torch.ones(2, 2), torch.zeros(2, 2))),
        ("vstack", _vstack, (torch.ones(1, 2), torch.zeros(1, 2))),
        ("cumsum", _cumsum, (torch.arange(6.0).reshape(2, 3),)),
        (
            "masked_fill",
            _masked_fill,
            (
                torch.arange(6.0).reshape(2, 3),
                torch.tensor([[True, False, False], [False, True, False]]),
            ),
        ),
        (
            "where",
            _where,
            (
                torch.tensor([[True, False], [False, True]]),
                torch.ones(2, 2),
                torch.zeros(2, 2),
            ),
        ),
        (
            "conv2d",
            _conv2d,
            (torch.ones(1, 2, 4, 4), torch.ones(3, 2, 3, 3)),
        ),
        (
            "embedding",
            _embedding,
            (torch.tensor([[0, 1], [2, 3]]), torch.arange(20.0).reshape(5, 4)),
        ),
        (
            "sdpa",
            _sdpa,
            (
                torch.arange(24.0).reshape(1, 2, 3, 4) / 10,
                torch.arange(24.0).reshape(1, 2, 3, 4) / 20,
                torch.arange(30.0).reshape(1, 2, 3, 5) / 30,
            ),
        ),
        (
            "linear",
            _linear,
            (
                torch.arange(6.0).reshape(2, 3),
                torch.arange(12.0).reshape(4, 3),
                torch.arange(4.0),
            ),
        ),
        (
            "layer_norm",
            _layer_norm,
            (
                torch.arange(6.0).reshape(2, 3),
                torch.ones(3),
                torch.zeros(3),
            ),
        ),
        (
            "max_values",
            _max_values,
            (torch.arange(6.0).reshape(2, 3),),
        ),
        (
            "split_recombine",
            _split_recombine,
            (torch.arange(12.0).reshape(2, 6),),
        ),
    ]


@pytest.mark.parametrize(("name", "reference", "inputs"), _cases())
def test_cpu_pipeline_preserves_reference_semantics(
    tmp_path: Path,
    name: str,
    reference: Callable[..., torch.Tensor],
    inputs: tuple[Any, ...],
) -> None:
    output = tmp_path / name
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

    actual = IRGraphExecutor(graph, verification_backend(graph["ir_kind"]))(
        *inputs
    )

    torch.testing.assert_close(actual, reference(*inputs), equal_nan=True)
    assert sorted(graph["source_input_indices"]) == sorted(
        operator.used_source_indices,
    )
    assert graph["outputs"]
    analysis = IRGraphAnalyzer(
        validator=ir_backend(graph["ir_kind"]).validate,
    ).analyze_graph(
        converted.path,
        output / "analysis",
        copy_graph=False,
        strict=True,
    )
    assert analysis is not None
    assert analysis["schema_version"] == 4


def test_extraction_tracks_only_tensor_inputs_used_by_reference(
    tmp_path: Path,
) -> None:
    unused = torch.zeros(2)

    def reference(value: torch.Tensor, scale: float, ignored: torch.Tensor):
        del ignored
        return value * scale, value + 1

    artifact = extract_operator_graph(
        reference,
        (torch.ones(2), 3.0, unused),
        device="cpu",
        output_dir=tmp_path,
        name="binding",
    )

    assert artifact.used_source_indices == (0,)
    assert [index for index, _ in artifact.source_inputs] == [0, 2]
    assert len(artifact.reference_outputs) == 2


def test_extraction_preserves_source_positions_when_use_order_differs(
    tmp_path: Path,
) -> None:
    def reference(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        return torch.sub(second, first)

    artifact = extract_operator_graph(
        reference,
        (torch.ones(2), torch.full((2,), 3.0)),
        device="cpu",
        output_dir=tmp_path,
        name="first-use-order",
    )

    assert artifact.used_source_indices == (0, 1)
    converted = convert_operator_graph(artifact, output_dir=tmp_path)
    graph = yaml.safe_load(converted.path.read_text())
    actual = IRGraphExecutor(graph, verification_backend(graph["ir_kind"]))(
        torch.ones(2),
        torch.full((2,), 3.0),
    )
    torch.testing.assert_close(actual, torch.full((2,), 2.0))


def test_torchview_preserves_repeated_use_of_one_source_input(
    tmp_path: Path,
) -> None:
    def reference(value: torch.Tensor) -> torch.Tensor:
        return value + value

    inputs = (torch.arange(6.0).reshape(2, 3),)
    operator = extract_operator_graph(
        reference,
        inputs,
        device="cpu",
        output_dir=tmp_path,
        name="repeated-source",
        extraction_kind=ExtractionKind.TORCHVIEW,
    )
    converted = convert_operator_graph(operator, output_dir=tmp_path)
    graph = yaml.safe_load(converted.path.read_text())
    add = next(
        layer
        for layer in graph["layers"].values()
        if (layer.get("semantic_op") or {}).get("target") == "add"
    )

    assert graph["source_input_indices"] == [0]
    assert add["tensor_names"]["inputs"][0] == add["tensor_names"]["inputs"][1]
    actual = IRGraphExecutor(graph, verification_backend(graph["ir_kind"]))(
        *inputs
    )
    torch.testing.assert_close(actual, reference(*inputs))


def test_torchview_preserves_internal_scalar_tensor_edges(
    tmp_path: Path,
) -> None:
    inputs = (
        torch.randn(2, 3),
        torch.softmax(torch.randn(2, 3), dim=1),
        4.0,
    )
    operator = extract_operator_graph(
        _scalar_tensor_chain,
        inputs,
        device="cpu",
        output_dir=tmp_path,
        name="scalar-tensor-chain",
        extraction_kind=ExtractionKind.TORCHVIEW,
    )
    converted = convert_operator_graph(operator, output_dir=tmp_path)
    graph = yaml.safe_load(converted.path.read_text())

    assert graph["source_input_indices"] == [0, 1]
    assert (
        len(
            [
                layer
                for layer in graph["layers"].values()
                if layer.get("type") == "start"
            ],
        )
        == 2
    )
    actual = IRGraphExecutor(graph, verification_backend(graph["ir_kind"]))(
        *inputs[:2],
    )
    torch.testing.assert_close(actual, _scalar_tensor_chain(*inputs))


def test_torchview_binds_tensor_keyword_arguments_to_source_positions(
    tmp_path: Path,
) -> None:
    inputs = (
        torch.randn(2, 3, 4, 4),
        torch.ones(3),
        torch.zeros(3),
        torch.zeros(3),
        torch.ones(3),
    )
    operator = extract_operator_graph(
        _batch_norm_with_tensor_kwargs,
        inputs,
        device="cpu",
        output_dir=tmp_path,
        name="tensor-keyword-arguments",
        extraction_kind=ExtractionKind.TORCHVIEW,
    )
    converted = convert_operator_graph(operator, output_dir=tmp_path)
    graph = yaml.safe_load(converted.path.read_text())

    assert graph["source_input_indices"] == [0, 1, 2, 3, 4]
    actual = IRGraphExecutor(graph, verification_backend(graph["ir_kind"]))(
        *inputs
    )
    torch.testing.assert_close(
        actual,
        _batch_norm_with_tensor_kwargs(*inputs),
    )


def test_torchview_linear_preserves_external_weight_and_bias(
    tmp_path: Path,
) -> None:
    inputs = (
        torch.randn(2, 3),
        torch.randn(4, 3),
        torch.randn(4),
    )
    operator = extract_operator_graph(
        _linear,
        inputs,
        device="cpu",
        output_dir=tmp_path,
        name="linear-external-weights",
        extraction_kind=ExtractionKind.TORCHVIEW,
    )
    converted = convert_operator_graph(operator, output_dir=tmp_path)
    graph = yaml.safe_load(converted.path.read_text())
    renamed_graph = yaml.safe_load(
        (tmp_path / "einsum_graph_renamed.yaml").read_text(),
    )

    assert graph["source_input_indices"] == [0, 1, 2]
    assert "source_node_remap" not in graph
    assert "_source_node_remap" not in graph
    assert "_source_node_remap" not in renamed_graph
    actual = IRGraphExecutor(graph, verification_backend(graph["ir_kind"]))(
        *inputs
    )
    torch.testing.assert_close(actual, _linear(*inputs))


@pytest.mark.parametrize(
    ("name", "reference", "inputs"),
    [
        (
            "max-value-slot",
            _max_values,
            (torch.arange(6.0).reshape(2, 3),),
        ),
        (
            "masked-fill-negative-inf",
            _masked_fill_negative_inf,
            (
                torch.arange(6.0).reshape(2, 3),
                torch.tensor(
                    [[True, False, False], [False, True, False]],
                ),
            ),
        ),
    ],
)
def test_torchview_preserves_exact_reduction_and_nonfinite_semantics(
    tmp_path: Path,
    name: str,
    reference: Callable[..., torch.Tensor],
    inputs: tuple[torch.Tensor, ...],
) -> None:
    operator = extract_operator_graph(
        reference,
        inputs,
        device="cpu",
        output_dir=tmp_path,
        name=name,
        extraction_kind=ExtractionKind.TORCHVIEW,
    )
    converted = convert_operator_graph(operator, output_dir=tmp_path)
    graph = yaml.safe_load(converted.path.read_text())

    actual = IRGraphExecutor(graph, verification_backend(graph["ir_kind"]))(
        *inputs
    )
    torch.testing.assert_close(
        actual,
        reference(*inputs),
        equal_nan=True,
    )


@pytest.mark.parametrize(
    ("name", "reference", "inputs", "target", "output_dtypes"),
    [
        (
            "split-outputs",
            _split_outputs,
            (torch.arange(12.0).reshape(2, 6),),
            "split",
            ["torch.float32"] * 3,
        ),
        (
            "topk-outputs",
            _topk_outputs,
            (torch.arange(16.0).reshape(2, 8),),
            "topk",
            ["torch.float32", "torch.int64"],
        ),
    ],
)
def test_torchview_preserves_independent_multi_output_slots(
    tmp_path: Path,
    name: str,
    reference: Callable[..., tuple[torch.Tensor, ...]],
    inputs: tuple[torch.Tensor, ...],
    target: str,
    output_dtypes: list[str],
) -> None:
    operator = extract_operator_graph(
        reference,
        inputs,
        device="cpu",
        output_dir=tmp_path,
        name=name,
        extraction_kind=ExtractionKind.TORCHVIEW,
    )
    converted = convert_operator_graph(operator, output_dir=tmp_path)
    graph = yaml.safe_load(converted.path.read_text())
    operation = next(
        layer
        for layer in graph["layers"].values()
        if (layer.get("semantic_op") or {}).get("target") == target
    )

    assert len(operation["tensor_names"]["outputs"]) == len(output_dtypes)
    assert operation["tensor_dtypes"]["outputs"] == output_dtypes
    assert len(graph["outputs"]) == len(output_dtypes)
    actual = IRGraphExecutor(graph, verification_backend(graph["ir_kind"]))(
        *inputs
    )
    expected = reference(*inputs)
    assert len(actual) == len(expected)
    for observed, wanted in zip(actual, expected, strict=True):
        torch.testing.assert_close(observed, wanted)


def test_extraction_rejects_non_tensor_reference_output(tmp_path: Path) -> None:
    with pytest.raises(
        RuntimeError,
        match="tensor reference inputs and outputs",
    ):
        extract_operator_graph(
            lambda value: int(value.sum()),
            (torch.ones(2),),
            device="cpu",
            output_dir=tmp_path,
            name="invalid",
        )


def test_canonical_extraction_preserves_explicit_backward_reference(
    tmp_path: Path,
) -> None:
    def reference(value: torch.Tensor, grad_output: torch.Tensor):
        tracked = value.clone().detach().requires_grad_()
        (tracked.square()).backward(grad_output)
        return tracked.grad

    inputs = (torch.arange(4.0), torch.ones(4))
    operator = extract_operator_graph(
        reference,
        inputs,
        device="cpu",
        output_dir=tmp_path,
        name="backward-reference",
        extraction_kind=ExtractionKind.MAKE_FX_REFERENCE,
    )
    converted = convert_operator_graph(
        operator,
        output_dir=tmp_path,
        ir_kind=IRKind.ATEN,
    )
    graph = yaml.safe_load(converted.path.read_text())

    assert graph["joint_graph"] is False
    assert any(
        layer.get("phase") == "reference" for layer in graph["layers"].values()
    )
    actual = IRGraphExecutor(graph, verification_backend(graph["ir_kind"]))(
        *inputs
    )
    torch.testing.assert_close(actual, reference(*inputs))


def test_make_fx_removes_only_unreachable_pure_allocations(
    tmp_path: Path,
) -> None:
    def dead_allocation(value: torch.Tensor) -> torch.Tensor:
        torch.empty_like(value)
        return value + 1

    operator = extract_operator_graph(
        dead_allocation,
        (torch.ones(4),),
        device="cpu",
        output_dir=tmp_path / "dead",
        name="dead-allocation",
        extraction_kind=ExtractionKind.MAKE_FX_REFERENCE,
    )
    graph = yaml.safe_load(operator.path.read_text())
    assert all(
        (layer.get("semantic_op") or {}).get("target") != "empty_like"
        for layer in graph["layers"].values()
    )
    convert_operator_graph(
        operator,
        output_dir=tmp_path / "dead",
        ir_kind=IRKind.ATEN,
    )

    reachable = extract_operator_graph(
        torch.empty_like,
        (torch.ones(4),),
        device="cpu",
        output_dir=tmp_path / "reachable",
        name="reachable-allocation",
        extraction_kind=ExtractionKind.MAKE_FX_REFERENCE,
    )
    with pytest.raises(ValueError, match="unsupported exact operation"):
        convert_operator_graph(
            reachable,
            output_dir=tmp_path / "reachable",
            ir_kind=IRKind.ATEN,
        )
