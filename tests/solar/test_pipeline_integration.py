from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn.functional as functional
import yaml

from solar.analysis.graph_analyzer import IRGraphAnalyzer
from solar.graph.extraction import extract_operator_graph
from solar.ir.conversion import convert_operator_graph
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
    )
    converted = convert_operator_graph(operator, output_dir=output)
    graph = yaml.safe_load(converted.path.read_text())

    actual = IRGraphExecutor(graph)(*inputs)

    torch.testing.assert_close(actual, reference(*inputs), equal_nan=True)
    assert sorted(graph["source_input_indices"]) == sorted(
        operator.used_source_indices,
    )
    assert graph["outputs"]
    analysis = IRGraphAnalyzer().analyze_graph(
        converted.path,
        output / "analysis",
        copy_graph=False,
        strict=True,
    )
    assert analysis is not None
    assert analysis["schema_version"] == 3


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
    )
    converted = convert_operator_graph(operator, output_dir=tmp_path)
    graph = yaml.safe_load(converted.path.read_text())

    assert graph["joint_graph"] is False
    assert any(
        layer.get("phase") == "reference" for layer in graph["layers"].values()
    )
    actual = IRGraphExecutor(graph)(*inputs)
    torch.testing.assert_close(actual, reference(*inputs))
