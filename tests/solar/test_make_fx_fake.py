# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Acceptance tests for the isolated static fake extraction pilot."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from scripts.internal.solar.l3a_measure import (
    projection_diff,
    semantic_projection,
)
from scripts.internal.solar.l3a_samples import get_focused_sample
from solar.graph.extraction import extract_operator_graph
from solar.graph.make_fx_fake import extract_operator_graph_fake_static
from solar.ir.contracts import IRPath


def _trace_documents(sample_id: str, tmp_path: Path) -> tuple[dict, dict]:
    sample = get_focused_sample(sample_id)
    inputs = sample.inputs(200)
    real = extract_operator_graph(
        sample.bound_reference(),
        inputs,
        device="cpu",
        output_dir=tmp_path / "real",
        name=sample_id,
        extraction_kind=IRPath.MAKE_FX_ATEN.extraction_kind,
    )
    fake = extract_operator_graph_fake_static(
        sample.bound_reference(),
        inputs,
        device="cpu",
        output_dir=tmp_path / "fake",
        name=sample_id,
    )
    return (
        yaml.safe_load(real.path.read_text()),
        yaml.safe_load(fake.path.read_text()),
    )


@pytest.mark.parametrize("sample_id", ["mutation_inplace", "scalar_constants"])
def test_fake_static_graph_matches_real_projection(
    sample_id: str, tmp_path: Path
) -> None:
    """O1: the fake graph keeps the tier-1 semantic projection identical."""
    real, fake = _trace_documents(sample_id, tmp_path)
    assert (
        projection_diff(semantic_projection(real), semantic_projection(fake))
        == []
    )


def test_fake_static_keeps_mutation_effects(tmp_path: Path) -> None:
    """O4: the mutation sample records schema-driven writes in both graphs."""
    real, fake = _trace_documents("mutation_inplace", tmp_path)

    def mutating_targets(document: dict) -> set[str]:
        return {
            layer["semantic_op"]["target"]
            for layer in document["layers"].values()
            if layer["semantic_op"].get("effects", {}).get("mutates")
        }

    assert mutating_targets(real) == mutating_targets(fake)
    assert mutating_targets(fake), "mutation effects were lost entirely"


def test_fake_static_rejects_data_dependent_branch(tmp_path: Path) -> None:
    """O4: converting fake scalars to bool must fail loudly, not trace."""
    sample = get_focused_sample("data_dependent_branch")
    with pytest.raises(Exception, match="(?i)data.?dependent|bool|scalar"):
        extract_operator_graph_fake_static(
            sample.bound_reference(),
            sample.inputs(200),
            device="cpu",
            output_dir=tmp_path / "fake",
            name="data_dependent_branch",
        )
    assert not (tmp_path / "fake" / "operator_graph.yaml").exists()


def test_projection_diff_reports_layer_and_signature_drift() -> None:
    """The oracle must localize layer, signature, and output differences."""
    base = {
        "layers": {
            "a": {
                "type": "add",
                "semantic_op": {"target": "add"},
                "tensor_shapes": {"inputs": [[2]], "outputs": [[2]]},
                "tensor_dtypes": {"inputs": ["Float"], "outputs": ["Float"]},
                "source_input_index": 0,
            }
        },
        "outputs": ["a"],
        "user_inputs": ["x"],
        "user_outputs": ["a"],
    }
    drift = dict(base)
    drift["layers"] = {
        "a": {**base["layers"]["a"], "type": "mul"},
    }
    assert projection_diff(base, drift) == ["/layers/a/type"]
    assert projection_diff(base, {**base, "outputs": ["b"]}) == ["/outputs"]
    assert projection_diff(base, base) == []
