from __future__ import annotations

import pytest

import solar
import solar.analysis
import solar.einsum
import solar.graph


def test_solar_public_api_exposes_only_atomic_pipeline() -> None:
    assert set(solar.__all__) == {
        "AnalysisFailure",
        "AnalysisRequest",
        "AnalysisResult",
        "ArtifactRef",
        "SolBound",
        "analyze",
    }
    with pytest.raises(AttributeError):
        getattr(solar, "PyTorchToEinsum")
    with pytest.raises(AttributeError):
        getattr(solar, "EinsumGraphAnalyzer")


def test_stage_packages_do_not_advertise_legacy_bypass_apis() -> None:
    assert solar.analysis.__all__ == []
    assert solar.einsum.__all__ == [
        "EinsumGraphArtifact",
        "convert_operator_graph",
    ]
    assert solar.graph.__all__ == [
        "OperatorGraphArtifact",
        "TensorSignature",
        "extract_operator_graph",
    ]
