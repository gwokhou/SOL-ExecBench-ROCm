from typing import Any

import pytest

from solar.analysis.resources import (
    RESOURCE_MODEL_VERSION,
    ResourceClassificationError,
    classify_layer_resources,
)


def _layer(
    target: str,
    *,
    input_shapes: list[list[int]] | None = None,
    output_shapes: list[list[int]] | None = None,
    input_dtypes: list[str] | None = None,
    output_dtypes: list[str] | None = None,
    kind: str = "call_function",
    semantic_extras: dict[str, Any] | None = None,
) -> dict[str, Any]:
    semantic = {"target": target, "kind": kind, **(semantic_extras or {})}
    return {
        "type": target,
        "semantic_op": semantic,
        "tensor_shapes": {
            "inputs": input_shapes or [],
            "outputs": output_shapes or [],
        },
        "tensor_dtypes": {
            "inputs": input_dtypes or [],
            "outputs": output_dtypes or [],
        },
    }


def _classify(layer: dict[str, Any], *, macs: int = 0, strict: bool = True):
    return classify_layer_resources(
        layer,
        macs=macs,
        fallback_precision="fp32",
        strict=strict,
    )


def test_einsum_rule_has_priority_over_target_specific_rules() -> None:
    result = _classify(
        _layer(
            "scaled_dot_product_attention",
            input_shapes=[[2, 4, 8], [2, 6, 8]],
            output_shapes=[[2, 4, 8]],
            input_dtypes=["fp16", "fp16"],
            output_dtypes=["fp16"],
            kind="einsum",
        ),
        macs=8,
    )

    assert result["work"] == {"mfma": {"fp16->fp32": 16}}
    assert result["formulas"] == ["2 * contraction_macs"]


def test_attention_rule_accounts_for_all_compute_resources() -> None:
    result = _classify(
        _layer(
            "scaled_dot_product_attention",
            input_shapes=[[2, 4, 8], [2, 6, 8]],
            output_shapes=[[2, 4, 8]],
            input_dtypes=["fp16", "fp16"],
            output_dtypes=["fp16"],
        )
    )

    assert result["work"] == {
        "mfma": {"fp16->fp32": 1536},
        "reduction": {"fp16": 80},
        "sfu": {"fp16": 48},
        "valu": {"fp16": 96},
    }


@pytest.mark.parametrize(
    ("layer", "reason"),
    [
        (_layer("reshape"), "metadata_or_alias_only"),
        (_layer("roll"), "memory_traffic_only"),
        (
            _layer(
                "to",
                input_shapes=[[4]],
                output_shapes=[[4]],
                input_dtypes=["fp32"],
                output_dtypes=["fp32"],
            ),
            "same_dtype_conversion_noop",
        ),
        (
            _layer(
                "sum",
                input_shapes=[[1]],
                output_shapes=[[1]],
                input_dtypes=["fp32"],
                output_dtypes=["fp32"],
            ),
            "degenerate_single_element_reduction",
        ),
    ],
)
def test_exempt_rules_return_stable_reason(layer: dict[str, Any], reason: str) -> None:
    assert _classify(layer) == {
        "model_version": RESOURCE_MODEL_VERSION,
        "work": {},
        "classification": "exempt",
        "exemption_reason": reason,
        "formulas": [],
    }


def test_atomic_rule_counts_source_values_and_uses_source_dtype() -> None:
    result = _classify(
        _layer(
            "index_add",
            input_shapes=[[2, 4], [2], [2, 4]],
            output_shapes=[[2, 4]],
            input_dtypes=["fp32", "int64", "fp16"],
            output_dtypes=["fp32"],
        )
    )

    assert result["work"] == {"atomic": {"fp16": 8}}


def test_softmax_rule_uses_recorded_reduction_dimension() -> None:
    result = _classify(
        _layer(
            "softmax",
            input_shapes=[[2, 4]],
            output_shapes=[[2, 4]],
            input_dtypes=["fp32"],
            output_dtypes=["fp32"],
            semantic_extras={"kwargs": {"dim": 1}},
        )
    )

    assert result["work"] == {
        "reduction": {"fp32": 12},
        "sfu": {"fp32": 8},
        "valu": {"fp32": 16},
    }


def test_unknown_operation_falls_back_to_macs_when_available() -> None:
    result = _classify(_layer("custom_contraction"), macs=3)

    assert result["work"] == {"mfma": {"fp32->fp32": 6}}


def test_unknown_operation_is_explicitly_unclassified_outside_strict_mode() -> None:
    result = _classify(_layer("custom_operation"), strict=False)

    assert result["classification"] == "unclassified"
    assert result["work"] == {}


def test_unknown_operation_is_rejected_in_strict_mode() -> None:
    with pytest.raises(ResourceClassificationError, match="custom_operation"):
        _classify(_layer("custom_operation"))
