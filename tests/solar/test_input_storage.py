"""Storage conditions are checked before replay can erase source aliases."""

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from scripts.internal.solar.base01_records import PROFILE
from solar.artifacts import sha256_file
from solar.contracts import ConversionRequest
from solar.errors import SourceInputBindingError
from solar.graph.storage import SourceStorage, capture_source_storage
from solar.ir.access_artifact import ACCESS_FILENAME, load_access_document
from solar.ir.contracts import IRPath
from solar.pipeline.readiness import (
    ConversionReadinessRequest,
    audit_conversion,
)


@pytest.mark.parametrize(
    "layout",
    ["contiguous", "transposed", "empty"],
    ids=["dense", "noncontiguous-injective", "zero-elements"],
)
def test_independent_source_layouts_round_trip(layout):
    value = torch.empty(4, 8)
    if layout == "transposed":
        value = value.t()
    elif layout == "empty":
        value = value[:0]
    record = capture_source_storage((value, "scalar", value.clone()))
    record.require_independent((0, 2))
    assert SourceStorage.model_validate_json(record.model_dump_json()) == record
    record.require_layout(
        capture_source_storage((value, None, value.clone())), (0, 2)
    )


@pytest.mark.parametrize(
    "layout",
    ["alias", "broadcast"],
    ids=["shared-arguments", "overlapping-elements"],
)
def test_storage_that_cannot_prove_independence_is_rejected(layout):
    value = torch.ones(8)
    inputs = (value, value[2:]) if layout == "alias" else (value[:1].expand(8),)
    record = capture_source_storage(inputs)
    with pytest.raises(SourceInputBindingError):
        record.require_independent(tuple(range(len(inputs))))


def test_changed_layout_cannot_reuse_trace_condition():
    expected = capture_source_storage((torch.ones(4, 4),))
    with pytest.raises(SourceInputBindingError, match="layout differs"):
        expected.require_layout(
            capture_source_storage((torch.ones(4, 4).t(),)), (0,)
        )


@pytest.mark.parametrize("ir_path", list(IRPath), ids=lambda item: item.value)
def test_aliases_appearing_only_in_verification_are_not_hidden_by_clone(
    tmp_path, ir_path
):
    def inputs(seed):
        value = torch.arange(8.0)
        return (value, value.clone()) if seed == 200 else (value, value)

    request = ConversionRequest(
        analysis_id="source-alias",
        reference=lambda x, y: x + y,
        input_factory=inputs,
        reference_name="test_input_storage",
        reference_sha256=sha256_file(Path(__file__)),
        ir_path=ir_path,
        trace_seed=200,
    )
    result = audit_conversion(
        ConversionReadinessRequest(
            conversion=request,
            architecture=PROFILE,
            output_dir=tmp_path / "conversion",
            require_verified_audit=False,
        )
    )
    assert not result.ready
    assert result.reason_code == SourceInputBindingError.reason_code
    assert result.failure_stage == "conversion_verification"


@pytest.mark.parametrize("ir_path", list(IRPath), ids=lambda item: item.value)
def test_trace_layout_is_captured_before_in_place_reference(tmp_path, ir_path):
    def reference(value):
        value[:2].add_(1)
        return value[1:]

    request = ConversionRequest(
        analysis_id="source-mutation",
        reference=reference,
        input_factory=lambda seed: (torch.arange(8.0),),
        reference_name="test_input_storage",
        reference_sha256=sha256_file(Path(__file__)),
        ir_path=ir_path,
    )
    request = replace(
        request,
        verification=replace(request.verification, verify_gradients=False),
    )
    output = tmp_path / "conversion"
    result = audit_conversion(
        ConversionReadinessRequest(
            conversion=request,
            architecture=PROFILE,
            output_dir=output,
            require_verified_audit=False,
        )
    )
    assert result.ready, result.to_dict()
    artifact = load_access_document(output / ir_path.graph_filename)
    assert artifact.source_storage is not None
    assert artifact.source_storage.inputs[0].shape == (8,)
    assert artifact.effect_bases
    import yaml

    attestation = yaml.safe_load(
        (output / "conversion-attestation.yaml").read_text()
    )
    subjects = {
        x["name"]: x["digest"]["sha256"] for x in attestation["subject"]
    }
    assert subjects[ACCESS_FILENAME] == sha256_file(output / ACCESS_FILENAME)
