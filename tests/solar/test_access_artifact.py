"""The access sidecar binds structured claims to exact executable IR bytes."""

import json
from pathlib import Path

import pytest
import torch
import yaml

from scripts.internal.solar.base01_records import PROFILE
from solar.artifacts import sha256_file
from solar.contracts import ConversionRequest
from solar.errors import SourceInputBindingError
from solar.graph.contracts import OPERATOR_GRAPH_FILENAME
from solar.ir.access_artifact import (
    ACCESS_FILENAME,
    MAX_ACCESS_ARTIFACT_BYTES,
    load_access_document,
)
from solar.ir.contracts import IRPath
from solar.pipeline.readiness import (
    ConversionReadinessRequest,
    audit_conversion,
)


@pytest.fixture(params=list(IRPath), ids=lambda item: item.value)
def access_files(request, tmp_path):
    path = request.param
    conversion = ConversionRequest(
        analysis_id="access-artifact",
        reference=lambda x: (x[:4].sum(), x[2:6].sum()),
        input_factory=lambda seed: (torch.arange(8.0),),
        reference_name="test_access_artifact",
        reference_sha256=sha256_file(Path(__file__)),
        ir_path=path,
    )
    output = tmp_path / "conversion"
    result = audit_conversion(
        ConversionReadinessRequest(
            conversion=conversion,
            architecture=PROFILE,
            output_dir=output,
            require_verified_audit=False,
        )
    )
    assert result.ready, result.to_dict()
    return output / path.graph_filename, output / ACCESS_FILENAME


def test_access_round_trip_and_exact_graph_binding(access_files):
    graph, path = access_files
    artifact = load_access_document(graph)
    artifact.require_graph(yaml.safe_load(graph.read_text()))
    assert artifact.source_indices
    assert artifact.transparent_layers
    assert artifact.graph_sha256 == sha256_file(graph)
    assert artifact.model_validate_json(path.read_bytes()) == artifact


@pytest.mark.parametrize(
    "damage",
    ["version", "missing-version", "graph", "operator", "map", "storage"],
    ids=[
        "foreign-schema",
        "absent-schema",
        "graph-bytes",
        "operator-bytes",
        "altered-access",
        "source-shape",
    ],
)
def test_damaged_access_claims_cannot_enter_analysis(access_files, damage):
    graph, path = access_files
    payload = json.loads(path.read_text())
    if damage == "version":
        payload = {
            "schema_version": -1,
            "source_storage": "invalid business fields",
        }
    elif damage == "missing-version":
        del payload["schema_version"]
    elif damage == "graph":
        graph.write_text(graph.read_text() + "\n# modified graph bytes\n")
    elif damage == "operator":
        operator = graph.with_name(OPERATOR_GRAPH_FILENAME)
        operator.write_text(operator.read_text() + "\n# changed provenance\n")
    elif damage == "map":
        payload["transparent_layers"] = []
    else:
        payload["source_storage"]["inputs"][0]["shape"] = [7]
    path.write_text(json.dumps(payload))
    with pytest.raises(
        (ValueError, SourceInputBindingError),
        match=r"schema version|SHA-256|claims disagree|logical source",
    ):
        load_access_document(graph).require_graph(
            yaml.safe_load(graph.read_text())
        )


def test_missing_operator_provenance_is_rejected(access_files):
    graph, _ = access_files
    graph.with_name(OPERATOR_GRAPH_FILENAME).unlink()
    with pytest.raises(FileNotFoundError):
        load_access_document(graph)


def test_access_byte_budget_precedes_json_parse(tmp_path):
    graph = tmp_path / "graph.yaml"
    path = tmp_path / ACCESS_FILENAME
    with path.open("wb") as stream:
        stream.seek(MAX_ACCESS_ARTIFACT_BYTES)
        stream.write(b"!")
    with pytest.raises(ValueError, match="byte budget"):
        load_access_document(graph)
