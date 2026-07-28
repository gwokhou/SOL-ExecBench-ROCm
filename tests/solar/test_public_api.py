from __future__ import annotations

import pytest

import solar
import solar.analysis
import solar.graph
import solar.ir.extended_einsum


def test_solar_public_api_exposes_only_atomic_pipeline() -> None:
    assert set(solar.__all__) == {
        "AnalysisFailure",
        "AnalysisRequest",
        "AnalysisResult",
        "ArtifactRef",
        "ConversionRequest",
        "ExtractionKind",
        "FormalProducerReadiness",
        "IRKind",
        "SolBound",
        "VerificationPolicy",
        "analyze",
        "architecture_profile_sha256",
        "formal_producer_readiness",
    }
    with pytest.raises(AttributeError):
        getattr(solar, "PyTorchToEinsum")  # noqa: B009 -- Assert absence
    with pytest.raises(AttributeError):
        getattr(solar, "IRGraphAnalyzer")  # noqa: B009 -- Assert absence


def test_stage_packages_do_not_advertise_legacy_bypass_apis() -> None:
    assert solar.analysis.__all__ == []
    assert solar.ir.extended_einsum.__all__ == [
        "convert_operator_graph",
        "lifecycle",
        "validate_extended_einsum_graph",
    ]
    assert solar.graph.__all__ == [
        "ExtractionKind",
        "OperatorGraphArtifact",
        "TensorSignature",
        "extract_operator_graph",
        "extraction_backend",
        "extraction_backends",
    ]
