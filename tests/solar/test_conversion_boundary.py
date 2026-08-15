"""Validation tests for the raw-to-normalized Torchview graph boundary."""

from __future__ import annotations

import pytest

from solar.ir.extended_einsum.torchview.converter_models import (
    ConversionError,
    normalize_conversion_graph,
)


def test_conversion_boundary_requires_layers_mapping() -> None:
    with pytest.raises(ConversionError, match="layers mapping"):
        normalize_conversion_graph({"model_name": "missing-layers"})


def test_conversion_boundary_rejects_malformed_shapes_and_connections() -> None:
    with pytest.raises(ConversionError, match="lists of integers"):
        normalize_conversion_graph(
            {"layers": {"add": {"input_shapes": [[2, "bad"]]}}}
        )
    with pytest.raises(ConversionError, match="connections must be a mapping"):
        normalize_conversion_graph(
            {"layers": {"add": {"connections": ["input"]}}}
        )


def test_conversion_boundary_normalizes_defaults_without_losing_fields() -> (
    None
):
    normalized = normalize_conversion_graph(
        {
            "schema_version": "operator.v1",
            "model_name": "demo",
            "layers": {
                "add": {
                    "type": "add",
                    "custom_provenance": "preserved",
                    "connections": {"inputs": ["x"]},
                }
            },
        }
    )
    layer = normalized["layers"]["add"]
    assert layer["connections"] == {"inputs": ["x"], "outputs": []}
    assert layer["input_shapes"] == []
    assert layer["module_args"] == {}
    assert dict(layer)["custom_provenance"] == "preserved"
