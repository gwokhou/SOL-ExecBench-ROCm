# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Backend fake-static tracing behavior (lineB production adoption)."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.internal.solar.l3a_samples import get_focused_sample
from solar.graph.extraction import extract_operator_graph
from solar.ir.contracts import IRPath


def _extract(sample_id: str, tmp_path: Path):
    sample = get_focused_sample(sample_id)
    return extract_operator_graph(
        sample.bound_reference(),
        sample.inputs(200),
        device="cpu",
        output_dir=tmp_path / sample_id,
        name=sample_id,
        extraction_kind=IRPath.MAKE_FX_ATEN.extraction_kind,
    )


@pytest.mark.parametrize("sample_id", ["mutation_inplace", "scalar_constants"])
def test_backend_extracts_focused_samples(
    sample_id: str, tmp_path: Path
) -> None:
    """Fake-static tracing extracts the focused graphs (lineB B2)."""
    artifact = _extract(sample_id, tmp_path)
    assert artifact.path.is_file()


def test_backend_records_mutation_effects(tmp_path: Path) -> None:
    """Schema mutation effects survive fake-static tracing."""
    import yaml

    artifact = _extract("mutation_inplace", tmp_path)
    graph = yaml.safe_load(artifact.path.read_text())
    mutating = [
        layer["semantic_op"]["target"]
        for layer in graph["layers"].values()
        if layer["semantic_op"].get("effects", {}).get("mutates")
    ]
    assert mutating, "mutation effects were lost"


def test_backend_rejects_data_dependent_branch(tmp_path: Path) -> None:
    """Converting fake scalars to bool must fail loudly, not trace."""
    sample = get_focused_sample("data_dependent_branch")
    with pytest.raises(Exception, match="(?i)data.?dependent|bool|scalar"):
        extract_operator_graph(
            sample.bound_reference(),
            sample.inputs(200),
            device="cpu",
            output_dir=tmp_path / "rejected",
            name="data_dependent_branch",
            extraction_kind=IRPath.MAKE_FX_ATEN.extraction_kind,
        )
    assert not (tmp_path / "rejected" / "operator_graph.yaml").exists()
