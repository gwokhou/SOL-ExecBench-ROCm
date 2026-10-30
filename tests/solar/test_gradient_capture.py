"""CPU replay checks for traced gradient boundaries, including backward references."""

from pathlib import Path
from typing import Any

import pytest
import torch
import yaml

from solar.errors import GradientVerificationError
from solar.graph.contracts import ExtractionKind
from solar.graph.extraction import extract_operator_graph
from solar.ir.contracts import IRKind
from solar.ir.conversion import convert_operator_graph
from solar.verification.contracts import VerificationPolicy
from solar.verification.executor import IRGraphExecutor
from solar.verification.gradients import verify_gradients
from solar.verification.registry import verification_backend


def normal(x):
    return x * x


def no_grad(x):
    with torch.no_grad():
        return x * x


def mixed(x):
    with torch.no_grad():
        y = x * x
    return y * x


def detached(x):
    return x.detach() * x


def nested(x):
    with torch.no_grad():
        y = x * x
        with torch.enable_grad():
            z = x * x
    return y * z


def backward(x):
    y = x.clone().detach().requires_grad_()
    (y * y).sum().backward()
    return y.grad


def backward_differentiable(x):
    y = x.clone().requires_grad_()
    return torch.autograd.grad((y * y).sum(), y, create_graph=True)[0]


def no_grad_view(x):
    with torch.no_grad():
        return x.view(-1)


def compile_reference(tmp_path: Path, reference: Any, inputs: tuple[Any, ...]):
    operator = extract_operator_graph(
        reference,
        inputs,
        device="cpu",
        output_dir=tmp_path,
        name="gradient_boundary",
        extraction_kind=ExtractionKind.MAKE_FX_REFERENCE,
    )
    converted = convert_operator_graph(
        operator, output_dir=tmp_path, ir_kind=IRKind.ATEN
    )
    graph = yaml.safe_load(converted.path.read_text())
    return graph, IRGraphExecutor(graph, verification_backend(graph["ir_kind"]))


@pytest.mark.parametrize(
    "reference",
    [
        normal,
        no_grad,
        mixed,
        detached,
        nested,
        backward,
        backward_differentiable,
    ],
    ids=lambda f: f.__name__,
)
@pytest.mark.parametrize(
    "trace_requires_grad", [False, True], ids=["plain-example", "grad-example"]
)
def test_boundary_preserves_values_and_strict_gradients(
    tmp_path, reference, trace_requires_grad
):
    inputs = (torch.tensor([1.0, 2.0], requires_grad=trace_requires_grad),)
    graph, executor = compile_reference(tmp_path, reference, inputs)
    torch.testing.assert_close(
        executor(*inputs), reference(*inputs), atol=0, rtol=0
    )
    result = verify_gradients(
        reference, executor, graph, inputs, VerificationPolicy(atol=0, rtol=0)
    )
    assert result["gradient_inputs_verified"] == 1


def test_no_grad_view_still_fails_strict_vjp(tmp_path):
    inputs = (torch.tensor([1.0, 2.0]),)
    graph, executor = compile_reference(tmp_path, no_grad_view, inputs)
    with pytest.raises(GradientVerificationError):
        verify_gradients(
            no_grad_view,
            executor,
            graph,
            inputs,
            VerificationPolicy(atol=0, rtol=0),
        )


def test_backward_reference_cpu_replay(tmp_path):
    import runpy

    path = (
        Path(__file__).resolve().parents[2]
        / "problems/AMD_AKA/instruction2triton/rmsnorm_bwd/reference.py"
    )
    reference = runpy.run_path(str(path))["run"]
    inputs = (
        torch.arange(1.0, 9.0).reshape(2, 4),
        torch.ones(4),
        torch.full((2, 4), 0.5),
    )
    graph, executor = compile_reference(tmp_path, reference, inputs)
    expected = reference(*inputs)
    actual = executor(*inputs)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert all(not value.requires_grad for value in actual)
    result = verify_gradients(
        reference, executor, graph, inputs, VerificationPolicy(atol=0, rtol=0)
    )
    assert result["gradient_inputs_verified"] == 3


def test_missing_internal_boundary_cannot_hide_behind_equal_values(tmp_path):
    def wrong(x):
        return x * x * x

    inputs = (torch.tensor([1.0, 2.0]),)
    graph, executor = compile_reference(tmp_path, wrong, inputs)
    torch.testing.assert_close(
        executor(*inputs), mixed(*inputs), atol=0, rtol=0
    )
    with pytest.raises(GradientVerificationError):
        verify_gradients(
            mixed, executor, graph, inputs, VerificationPolicy(atol=0, rtol=0)
        )


def test_blanket_output_detach_cannot_hide_missing_gradient(tmp_path):
    def wrong(x):
        return (x * x).detach()

    inputs = (torch.tensor([1.0, 2.0]),)
    graph, executor = compile_reference(tmp_path, wrong, inputs)
    torch.testing.assert_close(
        executor(*inputs), normal(*inputs), atol=0, rtol=0
    )
    with pytest.raises(GradientVerificationError, match="differentiability"):
        verify_gradients(
            normal, executor, graph, inputs, VerificationPolicy(atol=0, rtol=0)
        )


@pytest.mark.parametrize("mode", ["sum", "mean", "max"])
@pytest.mark.parametrize("include_last_offset", [False, True])
@pytest.mark.parametrize("padding_idx", [None, 1])
def test_embedding_schema_defaults_and_explicit_options(
    tmp_path, mode, include_last_offset, padding_idx
):
    def reference(indices, weights, offsets):
        return torch.nn.functional.embedding_bag(
            indices,
            weights,
            offsets,
            mode=mode,
            include_last_offset=include_last_offset,
            padding_idx=padding_idx,
        )

    inputs = (
        torch.tensor([0, 1, 2, 1]),
        torch.arange(18.0).reshape(6, 3),
        torch.tensor([0, 2, 4] if include_last_offset else [0, 2]),
    )
    graph, executor = compile_reference(tmp_path, reference, inputs)
    torch.testing.assert_close(
        executor(*inputs), reference(*inputs), atol=0, rtol=0
    )
    assert (
        verify_gradients(
            reference,
            executor,
            graph,
            inputs,
            VerificationPolicy(atol=0, rtol=0),
        )["gradient_inputs_verified"]
        == 1
    )


def test_no_grad_tuple_detaches_only_the_affected_output(tmp_path):
    def reference(x):
        with torch.no_grad():
            maximum, indices = torch.max(x, dim=0)
        return x * x, maximum, indices

    inputs = (torch.tensor([[1.0, 4.0], [3.0, 2.0]]),)
    graph, executor = compile_reference(tmp_path, reference, inputs)
    torch.testing.assert_close(
        executor(*inputs), reference(*inputs), atol=0, rtol=0
    )
    assert (
        verify_gradients(
            reference,
            executor,
            graph,
            inputs,
            VerificationPolicy(atol=0, rtol=0),
        )["gradient_inputs_verified"]
        == 1
    )


def test_no_grad_mutation_not_inferred_as_pure_detach(tmp_path):
    def reference(x):
        y = x.clone()
        with torch.no_grad():
            y.mul_(2)
        return y

    inputs = (torch.tensor([1.0, 2.0]),)
    graph, executor = compile_reference(tmp_path, reference, inputs)
    torch.testing.assert_close(
        executor(*inputs), reference(*inputs), atol=0, rtol=0
    )
    with pytest.raises(GradientVerificationError):
        verify_gradients(
            reference,
            executor,
            graph,
            inputs,
            VerificationPolicy(atol=0, rtol=0),
        )
