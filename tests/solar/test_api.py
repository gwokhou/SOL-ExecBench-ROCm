from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

import solar.api as api
from solar.api import AnalysisFailure, AnalysisRequest, AnalysisResult
from solar.einsum.conversion import EinsumGraphArtifact
from solar.graph.extraction import OperatorGraphArtifact
from solar.analysis.orojenesis import OrojenesisError
from solar.verification import VerificationError


class _Profile:
    def to_dict(self):
        return {"name": "test", "gfx_target": "gfx1200"}


def _request(output: Path) -> AnalysisRequest:
    return AnalysisRequest(
        analysis_id="problem:workload",
        reference=lambda value: value,
        input_factory=lambda seed: (seed,),
        reference_name="definition.json#reference",
        reference_sha256="a" * 64,
        architecture="RX_9060_XT",
        output_dir=output,
    )


def test_analyze_publishes_only_complete_atomic_artifact_set(tmp_path, monkeypatch):
    output = tmp_path / "result"
    monkeypatch.setattr(api.ArchitectureProfile, "load", lambda value: _Profile())

    def extract(reference, inputs, *, device, output_dir, name):
        del reference, inputs, device, name
        root = Path(output_dir)
        operator = root / "operator_graph.yaml"
        operator.write_text("layers: {}\n")
        return OperatorGraphArtifact(operator, (), (), ())

    def convert(operator, *, output_dir):
        del operator
        einsum = Path(output_dir) / "einsum_graph.yaml"
        einsum.write_text("layers: {}\n")
        return EinsumGraphArtifact(einsum)

    def verify(**kwargs):
        Path(kwargs["output_path"]).write_text("predicate: passed\n")

    analysis = {
        "total": {"lower_bound_seconds": 0.001, "compute_resource": "mfma"},
        "metadata": {"bound_kind": "capacity_constrained_tile_aware_v1"},
    }
    monkeypatch.setattr(api, "extract_operator_graph", extract)
    monkeypatch.setattr(api, "convert_operator_graph", convert)
    monkeypatch.setattr(api, "verify_callable_conversion", verify)
    monkeypatch.setattr(api, "_run_analysis", lambda request, profile, root: analysis)

    result = api.analyze(_request(output))

    assert isinstance(result, AnalysisResult)
    assert result.bound.seconds == 0.001
    assert {path.name for path in output.iterdir()} == {
        "operator_graph.yaml",
        "einsum_graph.yaml",
        "conversion-attestation.yaml",
        "solar-analysis.yaml",
        "manifest.yaml",
    }
    manifest = yaml.safe_load((output / "manifest.yaml").read_text())
    assert "candidate_runtime" not in manifest
    assert "score" not in manifest
    assert manifest["analysis_contract"]["precision"] == "fp16"


def test_analyze_failure_leaves_no_partial_output(tmp_path, monkeypatch):
    output = tmp_path / "result"
    monkeypatch.setattr(api.ArchitectureProfile, "load", lambda value: _Profile())
    monkeypatch.setattr(
        api,
        "extract_operator_graph",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("unsupported")),
    )

    result = api.analyze(_request(output))

    assert isinstance(result, AnalysisFailure)
    assert result.stage == "graph_extraction"
    assert result.reason_code == "graph_extraction_failed"
    assert not output.exists()


def test_conversion_failure_has_its_own_stable_stage(tmp_path, monkeypatch):
    output = tmp_path / "result"
    monkeypatch.setattr(api.ArchitectureProfile, "load", lambda value: _Profile())

    def extract(reference, inputs, *, device, output_dir, name):
        del reference, inputs, device, name
        operator = Path(output_dir) / "operator_graph.yaml"
        operator.write_text("layers: {}\n")
        return OperatorGraphArtifact(operator, (), (), ())

    monkeypatch.setattr(api, "extract_operator_graph", extract)
    monkeypatch.setattr(
        api,
        "convert_operator_graph",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("unsupported op")),
    )

    result = api.analyze(_request(output))

    assert isinstance(result, AnalysisFailure)
    assert result.stage == "einsum_conversion"
    assert result.reason_code == "einsum_conversion_failed"
    assert not output.exists()


def test_packaged_profile_blocks_formal_analysis_without_audit_artifact(tmp_path):
    result = api.analyze(_request(tmp_path / "result"))

    assert isinstance(result, AnalysisFailure)
    assert result.stage == "architecture"
    assert result.reason_code == "architecture_failed"
    assert "audit evidence unavailable" in result.message


@pytest.mark.parametrize(
    "changes",
    [
        {"analysis_id": ""},
        {"reference_name": ""},
        {"reference_sha256": "A" * 64},
        {"reference_sha256": "a" * 63},
        {"atol": -1},
        {"rtol": float("nan")},
        {"max_error_cap": float("inf")},
        {"required_matched_ratio": 1.1},
    ],
)
def test_analysis_request_rejects_invalid_contract_fields(tmp_path, changes):
    values: dict[str, Any] = {
        "analysis_id": "analysis",
        "reference": lambda value: value,
        "input_factory": lambda seed: (seed,),
        "reference_name": "reference",
        "reference_sha256": "a" * 64,
        "architecture": {},
        "output_dir": tmp_path / "result",
    }
    values.update(changes)
    with pytest.raises(ValueError):
        AnalysisRequest(**values)


def test_analyze_refuses_to_overwrite_existing_output(tmp_path):
    output = tmp_path / "result"
    output.mkdir()
    result = api.analyze(_request(output))
    assert isinstance(result, AnalysisFailure)
    assert result.stage == "prepare"
    assert result.reason_code == "output_exists"


def test_bound_and_reason_code_helpers_fail_closed():
    valid = api._extract_bound(
        {
            "total": {"lower_bound_seconds": 0, "compute_resource": None},
            "metadata": {"bound_kind": "capacity_constrained_tile_aware_v1"},
        }
    )
    assert valid.seconds == 0
    assert valid.limiting_resource is None
    for seconds in (None, -1, float("nan")):
        with pytest.raises(ValueError, match="finite lower bound"):
            api._extract_bound(
                {
                    "total": {"lower_bound_seconds": seconds},
                    "metadata": {"bound_kind": "capacity_constrained_tile_aware_v1"},
                }
            )
    with pytest.raises(ValueError, match="non-formal"):
        api._extract_bound(
            {
                "total": {"lower_bound_seconds": 1},
                "metadata": {"bound_kind": "roofline"},
            }
        )
    assert api._reason_code("formal_analysis", OrojenesisError("missing")) == (
        "toolchain_unavailable"
    )
    assert api._reason_code("conversion_verification", VerificationError("bad")) == (
        "conversion_not_proven"
    )
    assert api._reason_code("graph_extraction", RuntimeError()) == (
        "graph_extraction_failed"
    )
