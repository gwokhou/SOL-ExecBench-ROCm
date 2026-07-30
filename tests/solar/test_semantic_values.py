from __future__ import annotations

import pytest
import torch

from solar.verification.semantic_values import (
    ATEN_VALUE_POLICY,
    EXTENDED_VALUE_POLICY,
    SemanticValueDecodeError,
    SemanticValueErrorKind,
    decode_semantic_value,
)


def test_shared_decoder_recurses_through_semantic_containers() -> None:
    operand = object()
    encoded = [
        {"tensor": 0},
        {"slice": [{"value": 1}, {"value": 4}, {"value": 2}]},
        {"value": "__ellipsis__"},
    ]

    decoded = decode_semantic_value(
        encoded,
        [operand],
        policy=ATEN_VALUE_POLICY,
    )

    assert decoded[0] is operand
    assert decoded[1] == slice(1, 4, 2)
    assert decoded[2] is Ellipsis


def test_aten_policy_decodes_bare_memory_formats() -> None:
    assert (
        decode_semantic_value(
            "preserve_format",
            (),
            policy=ATEN_VALUE_POLICY,
        )
        is torch.preserve_format
    )
    assert (
        decode_semantic_value(
            {"value": "contiguous_format"},
            (),
            policy=ATEN_VALUE_POLICY,
        )
        is torch.contiguous_format
    )


def test_extended_policy_prefers_literal_discriminator() -> None:
    assert (
        decode_semantic_value(
            {"literal": 3, "value": 4},
            (),
            policy=EXTENDED_VALUE_POLICY,
        )
        == 3
    )


@pytest.mark.parametrize(
    ("encoded", "kind"),
    (
        ({"tensor": 1}, SemanticValueErrorKind.MISSING_TENSOR),
        ({"dtype": "not_a_dtype"}, SemanticValueErrorKind.INVALID_DTYPE),
        ({"layout": "not_a_layout"}, SemanticValueErrorKind.INVALID_LAYOUT),
        ({"unknown": 1}, SemanticValueErrorKind.INVALID_VALUE),
    ),
    ids=(
        "missing-tensor",
        "invalid-dtype",
        "invalid-layout",
        "unknown-discriminator",
    ),
)
def test_shared_decoder_reports_typed_failures(
    encoded: dict[str, object],
    kind: SemanticValueErrorKind,
) -> None:
    with pytest.raises(SemanticValueDecodeError) as captured:
        decode_semantic_value(
            encoded,
            (),
            policy=EXTENDED_VALUE_POLICY,
        )

    assert captured.value.kind is kind
