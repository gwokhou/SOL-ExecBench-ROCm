"""Regression cases for R accounting, without device access or GPU receipts."""

from pathlib import Path

import pytest
import torch
import yaml
from torch.nn import functional

from scripts.internal.solar.base01_records import PROFILE
from solar.analysis.graph_analyzer import IRGraphAnalyzer
from solar.analysis.resources import classify_layer_resources
from solar.api import ConversionRequest, VerificationPolicy
from solar.ir.contracts import IRPath
from solar.ir.extended_einsum.torchview.semantics import (
    build_semantic_operation,
)
from solar.ir.registry import ir_backend
from solar.pipeline.readiness import (
    ConversionReadinessRequest,
    audit_conversion,
)


@pytest.mark.parametrize(
    ("source", "destination", "expected"),
    [
        ("int64", "int32", {"conversion": {"integer->integer": 6}}),
        ("int32", "int64", {"conversion": {"integer->integer": 6}}),
        ("uint8", "int8", {"conversion": {"integer->integer": 6}}),
        ("int32", "torch.int32", {}),
        ("float32", "fp32", {}),
    ],
)
def test_conversion_identity_precedes_resource_mode(
    source, destination, expected
):
    layer = {
        "semantic_op": {"kind": "operation", "target": "to"},
        "tensor_shapes": {"inputs": [[2, 3]], "outputs": [[2, 3]]},
        "tensor_dtypes": {"inputs": [source], "outputs": [destination]},
    }
    result = classify_layer_resources(
        layer, macs=0, fallback_precision="fp32", strict=True
    )
    assert result["work"] == expected
    assert (result["exemption_reason"] == "same_dtype_conversion_noop") == (
        not expected
    )


@pytest.mark.parametrize(
    ("target", "attributes", "atomic", "mutates", "opaque"),
    [
        ("where", {}, False, [], False),
        ("index_put", {}, True, [], False),
        ("relu", {"inplace": {"value": False}}, False, [], False),
        ("relu", {"inplace": {"value": True}}, False, [0], False),
        ("elu", {"inplace": {"value": True}}, False, [0], False),
        ("relu", {}, False, [], False),
        ("batch_norm", {}, False, [], True),
        ("conv2d", {}, False, [], False),
    ],
)
def test_pure_native_effects_keep_mutation_boundaries(
    target, attributes, atomic, mutates, opaque
):
    layer = {
        "type": target,
        "tensor_names": {
            "inputs": ["x"] if target in {"relu", "elu"} else ["x", "y", "z"],
            "outputs": ["out"],
        },
        "module_args": {"call_kwargs": attributes},
    }
    effects = build_semantic_operation(layer)["effects"]
    assert effects["atomic"] is atomic
    assert effects["mutates"] == mutates
    assert effects["opaque_library_call"] is opaque


def _reference(case):
    if case == "mlp":
        return lambda x, w, b: functional.relu(functional.linear(x, w, b))
    if case == "addmm-zero-bias":
        return lambda x, w, b: torch.addmm(b, x, w.t(), beta=0)
    if case == "ones-like":
        return lambda x, _w, _b: torch.ones_like(x)
    if case == "pointwise-conv":
        return lambda x, w, b: functional.elu(functional.conv2d(x, w, b))
    if case == "strided-conv":
        return lambda x, w, b: functional.conv2d(x, w, b, 2)
    if case == "quant":

        def quant(x, _w, _b):
            value = x.float()
            scale = value.abs().amax(dim=-1, keepdim=True) / 127
            scale = torch.where(scale == 0, torch.ones_like(scale), scale)
            return torch.clamp(value / scale, -128, 127).to(torch.int8), scale

        return quant
    if case == "where-scalar":
        return lambda x, w, _b: torch.where(x == 0, w, 0.0)
    if case == "where-two-scalars":
        return lambda x, _w, _b: torch.where(x == 0, 1.0, 0.0)
    if case == "conditional-external":
        return lambda x, w, b: torch.where(x == 0, w, b)
    if case == "gather":
        return lambda x, w, _b: torch.gather(x, 1, w.to(torch.int64))
    raise AssertionError(case)


def _inputs(case):
    if "conv" in case:
        return torch.ones(1, 2, 4, 4), torch.ones(3, 2, 1, 1), torch.ones(3)
    if case in {"mlp", "addmm-zero-bias"}:
        return torch.ones(2, 4), torch.ones(3, 4), torch.ones(3)
    if case == "gather":
        return torch.ones(2, 4), torch.zeros(2, 1), torch.ones(1)
    return torch.ones(2, 4), torch.ones(2, 4), torch.ones(2, 4)


def _analyze(tmp_path: Path, path: IRPath, case: str):
    from solar.artifacts import sha256_file

    request = ConversionRequest(
        analysis_id=case,
        reference=_reference(case),
        input_factory=lambda _seed: _inputs(case),
        reference_name=case,
        reference_sha256=sha256_file(Path(__file__)),
        ir_path=path,
        verification=VerificationPolicy(
            atol=1e-5,
            rtol=1e-5,
            device="cpu",
            preserved_input_indices=(1,) if case == "gather" else (),
        ),
    )
    root = tmp_path / case / path.value
    ready = audit_conversion(
        ConversionReadinessRequest(
            conversion=request,
            architecture=PROFILE,
            output_dir=root,
            require_verified_audit=False,
        )
    )
    existing_refusals = {
        "addmm-zero-bias": "native_attribute_unsupported",
        "ones-like": "reference_output_binding_failed",
    }
    if path is IRPath.TORCHVIEW_EXTENDED_EINSUM and case in existing_refusals:
        assert not ready.ready
        assert ready.reason_code == existing_refusals[case]
        return None, None
    assert ready.ready, ready.to_dict()
    graph = yaml.safe_load((root / path.graph_filename).read_text())
    result = IRGraphAnalyzer(
        validator=ir_backend(path.ir_kind).validate
    ).analyze_graph(
        root / path.graph_filename,
        root / "analysis",
        strict=True,
        architecture=PROFILE,
        copy_graph=False,
    )
    assert result is not None
    return graph, result


@pytest.mark.parametrize("path", list(IRPath))
@pytest.mark.parametrize(
    ("case", "available"),
    [
        ("mlp", True),
        ("addmm-zero-bias", False),
        ("ones-like", True),
        ("pointwise-conv", True),
        ("strided-conv", False),
        ("quant", True),
        ("conditional-external", False),
        ("where-scalar", False),
        ("where-two-scalars", True),
        ("gather", False),
    ],
)
def test_read_contracts_reach_analysis_without_relaxing_unknowns(
    tmp_path, path, case, available
):
    graph, result = _analyze(tmp_path, path, case)
    if result is None:
        assert path is IRPath.TORCHVIEW_EXTENDED_EINSUM
        assert case in {"addmm-zero-bias", "ones-like"}
        return
    assert graph is not None
    reads = result["metadata"]["external_reads"]
    assert (reads["quality"] != "unavailable") is available, (
        reads,
        [
            (name, layer.get("semantic_op"))
            for name, layer in graph["layers"].items()
        ],
    )
    for name in ("unfused", "ideal_fused"):
        view = result["metadata"]["views"][name]
        assert (view["status"] == "available") is available
        assert (view["seconds"] is not None) is available
    if case == "mlp":
        assert result["total"]["macs"] == 24
        # Fused addmm can initialize its accumulator with bias. A separate
        # graph add is counted separately, without inventing mandatory VALU
        # instructions inside the fused operator.
        assert result["total"]["resource_work"]["valu"]["fp32"] == (
            12 if path is IRPath.TORCHVIEW_EXTENDED_EINSUM else 6
        )
    if case == "pointwise-conv":
        assert reads["bytes"] == (32 + 6 + 3) * 4
    if case == "ones-like":
        assert reads["bytes"] == 0
    if case == "quant":
        assert reads["bytes"] == 2 * 4 * 4


def test_single_argument_where_does_not_become_static_selection():
    from solar.ir.extended_einsum.torchview.semantics import (
        SemanticGraphError,
        _validate_native_operation,
    )

    semantic = build_semantic_operation(
        {
            "type": "where",
            "tensor_names": {"inputs": ["mask"], "outputs": ["indices"]},
        }
    )
    with pytest.raises(SemanticGraphError, match="three where operands"):
        _validate_native_operation(
            "where", semantic, {"inputs": ["mask"], "outputs": ["indices"]}
        )


def test_where_out_keeps_four_tensor_operands_and_effect_boundary():
    from solar.ir.extended_einsum.torchview.semantics import (
        _validate_native_operation,
    )

    names = {"inputs": ["mask", "left", "right", "out"], "outputs": ["result"]}
    semantic = build_semantic_operation(
        {
            "type": "where",
            "tensor_names": names,
            "module_args": {
                "call_arguments": [{"tensor": 0}, {"tensor": 1}, {"tensor": 2}],
                "call_kwargs": {"out": {"tensor": 3}},
            },
        }
    )
    _validate_native_operation("where-out", semantic, names)
    assert semantic["effects"]["mutates"] == [3]
    assert semantic["effects"]["aliases"] == [{"output": 0, "input": 3}]
    assert semantic["effects"]["atomic"] is False


@pytest.mark.parametrize(
    ("target", "inputs", "operand", "known"),
    [
        ("scalar_tensor", [], {"value": 1.0}, True),
        ("scalar_tensor", ["x"], {"value": 1.0}, False),
        ("scalar_tensor", [], {"tensor": 0}, False),
        ("unknown_factory", [], {"value": 1.0}, False),
    ],
)
def test_literal_scalar_read_contract_is_narrow(target, inputs, operand, known):
    from solar.analysis.external_reads import _blocking_reason
    from solar.ir.access import AccessReason
    from solar.ir.access_artifact import AccessDocument
    from solar.ir.contracts import IRKind

    access = AccessDocument(
        graph_sha256="0" * 64,
        operator_sha256="0" * 64,
        ir_kind=IRKind.ATEN,
        source_storage=None,
        source_indices={},
        tensors={},
        reasons={},
        transparent_layers=frozenset(),
        effect_bases=frozenset(),
    )
    layer = {
        "semantic_op": {
            "kind": "aten",
            "target": target,
            "arguments": [operand],
        },
        "tensor_names": {"inputs": inputs, "outputs": ["out"]},
    }
    assert _blocking_reason(layer, access, {}, {}) == (
        None if known else AccessReason.UNKNOWN_READS
    )
