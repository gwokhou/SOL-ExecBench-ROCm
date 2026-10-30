"""Independent arithmetic/traffic oracles for full-review counterexamples."""

from pathlib import Path

import pytest
import torch
from torch.nn import functional

from scripts.internal.solar.base01_records import PROFILE
from solar.analysis.graph_analyzer import IRGraphAnalyzer
from solar.analysis.graph_context import AnalysisJob
from solar.analysis.resources import mandatory_mfma_macs
from solar.api import ConversionRequest, VerificationPolicy
from solar.ir.contracts import IRPath
from solar.ir.extended_einsum.operations.handlers.base import (
    compute_cost_from_equation,
)
from solar.ir.registry import ir_backend
from solar.pipeline.readiness import (
    ConversionReadinessRequest,
    audit_conversion,
)
from solar.rocm.architecture import ArchitectureProfile
from solar.types import TensorShapes


def _alias_write(x):
    y = x + 1
    alias = y.view(-1)
    y.add_(1)
    return alias * 2


CASES = {
    "local_write": (_alias_write, lambda _: (torch.ones(2, 3),)),
    "ones_like": (lambda x: torch.ones_like(x), lambda _: (torch.ones(2, 4),)),
    "broadcast_mm": (
        lambda x, w: x @ w,
        lambda _: (torch.ones(1, 2, 3), torch.ones(4, 3, 5)),
    ),
    "conv_transpose": (
        lambda x, w: functional.conv_transpose2d(x, w, stride=2),
        lambda _: (torch.ones(1, 1, 2, 2), torch.ones(1, 1, 3, 3)),
    ),
    "mixed_add": (
        lambda x, y: x + y,
        lambda _: (torch.ones(2, 3, dtype=torch.int32), torch.ones(2, 3)),
    ),
    "mixed_reverse": (
        lambda x, y: y + x,
        lambda _: (torch.ones(2, 3, dtype=torch.int32), torch.ones(2, 3)),
    ),
}


@pytest.mark.parametrize("ir_path", list(IRPath))
@pytest.mark.parametrize("name", list(CASES))
def test_replayed_work_has_exact_operation_or_traffic_count(
    tmp_path: Path, ir_path, name
):
    reference, factory = CASES[name]
    conversion = ConversionRequest(
        analysis_id=name,
        reference=reference,
        input_factory=factory,
        reference_name=name,
        reference_sha256="a" * 64,
        ir_path=ir_path,
        verification=VerificationPolicy(atol=1e-5, rtol=1e-5, device="cpu"),
    )
    ready = audit_conversion(
        ConversionReadinessRequest(
            conversion=conversion,
            architecture=PROFILE,
            output_dir=tmp_path / "ready",
            require_verified_audit=False,
        )
    )
    if name == "ones_like" and ir_path is IRPath.TORCHVIEW_EXTENDED_EINSUM:
        assert not ready.ready
        assert ready.reason_code == "reference_output_binding_failed"
        return
    assert ready.ready, ready.message
    analyzer = IRGraphAnalyzer(validator=ir_backend(ir_path.ir_kind).validate)
    intrinsic = analyzer.prepare_intrinsic(
        AnalysisJob(
            graph_path=tmp_path / "ready" / ir_path.graph_filename,
            output_dir=tmp_path / "counts",
            precision="fp32",
            copy_graph=False,
            strict=True,
            architecture=PROFILE,
            orojenesis_runner=None,
            require_orojenesis=False,
        )
    )
    assert intrinsic is not None
    result = analyzer.project_intrinsic(
        intrinsic,
        output_dir=tmp_path / "dense",
        architecture=ArchitectureProfile.load(PROFILE),
        write_output=False,
    )
    total = result["total"]
    if name == "local_write":
        assert total["resource_work"] == {"valu": {"fp32": 18}}
        assert total["num_orphaned_layers"] == 0
    elif name == "ones_like":
        assert total["unfused_bytes"] == 32
    elif name in {"mixed_add", "mixed_reverse"}:
        assert total["resource_work"] == {"valu": {"fp32": 6}}
    else:
        assert total["macs"] == (120 if name == "broadcast_mm" else 36)


@pytest.mark.parametrize("batch, expected", [(4, 120), (0, 0), (1, 30)])
def test_einsum_axis_broadcast_uses_all_operand_sizes(batch, expected):
    assert (
        compute_cost_from_equation(
            "BMK,BKN->BMN",
            TensorShapes(
                inputs=[[1, 2, 3], [batch, 3, 5]],
                outputs=[[batch, 2, 5]],
            ),
        )
        == expected
    )


def test_einsum_rejects_conflicting_non_singleton_dimensions():
    with pytest.raises(ValueError, match="incompatible broadcast"):
        compute_cost_from_equation(
            "MK,KN->MN", TensorShapes(inputs=[[2, 3], [4, 5]], outputs=[[2, 5]])
        )


@pytest.mark.parametrize(
    "stride, padding, dilation, output_padding",
    [(1, 0, 1, 0), (2, 0, 1, 1), (2, 1, 2, 0), (1, 2, 1, 0)],
)
@pytest.mark.parametrize("dialect", ["native", "aten"])
def test_transposed_convolution_counts_only_contributions_in_output(
    stride, padding, dilation, output_padding, dialect
):
    size, kernel, groups = 4, 3, 2
    output = (
        (size - 1) * stride
        - 2 * padding
        + dilation * (kernel - 1)
        + output_padding
        + 1
    )
    pairs = sum(
        0 <= i * stride - padding + k * dilation < output
        for i in range(size)
        for k in range(kernel)
    )
    semantic = {
        "attributes": {
            "stride": stride,
            "padding": padding,
            "dilation": dilation,
            "groups": groups,
        }
    }
    if dialect == "aten":
        semantic = {
            "exact_target": "convolution",
            "arguments": [
                {"tensor": 0},
                {"tensor": 1},
                {"value": None},
                [stride],
                [padding],
                [dilation],
                True,
                [output_padding],
                groups,
            ],
        }
    assert (
        mandatory_mfma_macs(
            "conv_transpose1d",
            [[2, 4, size], [4, 3, kernel]],
            [[2, 6, output]],
            semantic,
        )
        == 2 * 4 * 3 * pairs
    )


@pytest.mark.parametrize("target", ["eq", "add"])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("scalar", [False, True])
def test_binary_mode_uses_data_promotion_not_output_cast_or_order(
    target, reverse, scalar
):
    from solar.analysis.resources import classify_layer_resources

    dtypes = ["torch.float64" if scalar else "torch.int32", "torch.float32"]
    shapes = [[], [3]] if scalar else [[3], [3]]
    if reverse:
        dtypes.reverse()
        shapes.reverse()
    layer = {
        "type": target,
        "semantic_op": {
            "kind": "operation",
            "target": target,
            "arguments": [{"tensor": 0}, {"tensor": 1}],
            "kwargs": {"out": {"tensor": 2}} if target == "add" else {},
            "effects": {"mutates": [2]} if target == "add" else {},
        },
        "tensor_shapes": {
            "inputs": shapes + ([[3]] if target == "add" else []),
            "outputs": [[3]],
        },
        "tensor_dtypes": {
            "inputs": dtypes + (["torch.float16"] if target == "add" else []),
            "outputs": ["torch.float16" if target == "add" else "torch.bool"],
        },
    }
    result = classify_layer_resources(
        layer, macs=0, fallback_precision="fp32", strict=True
    )
    assert result["work"] == {"valu": {"fp32": 3}}


@pytest.mark.parametrize(
    "operands,dtypes,expected",
    [
        (
            [{"tensor": 0}, {"tensor": 1}],
            ["torch.float16", "torch.bfloat16"],
            "fp32",
        ),
        ([{"tensor": 0}, {"value": 0.5}], ["torch.int32"], "fp32"),
    ],
)
def test_binary_mode_preserves_bfloat_promotion_and_literal_category(
    operands, dtypes, expected
):
    from solar.analysis.resources import classify_layer_resources

    layer = {
        "type": "eq",
        "semantic_op": {
            "kind": "operation",
            "target": "eq",
            "arguments": operands,
        },
        "tensor_shapes": {"inputs": [[3]] * len(dtypes), "outputs": [[3]]},
        "tensor_dtypes": {"inputs": dtypes, "outputs": ["torch.bool"]},
    }
    result = classify_layer_resources(
        layer, macs=0, fallback_precision="fp32", strict=True
    )
    assert result["work"] == {"valu": {expected: 3}}
