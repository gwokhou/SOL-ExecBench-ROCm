"""Explicit fusion validation API."""

from .io import canonical_json_bytes, sha256_payload
from .models import (
    FUSION_VALIDATION_SCHEMA_VERSION,
    FusionCapacityStatus,
    FusionPerformanceStatus,
    FusionSignature,
    FusionValidationArtifact,
    FusionValidationCase,
    KernelResourceEvidence,
    PerformanceEvidence,
)
from .parsing import fusion_validation_from_dict
from .performance import performance_from_rounds
from .resources import kernel_resource_from_code_object

__all__ = [
    "FUSION_VALIDATION_SCHEMA_VERSION",
    "FusionCapacityStatus",
    "FusionPerformanceStatus",
    "FusionSignature",
    "FusionValidationArtifact",
    "FusionValidationCase",
    "KernelResourceEvidence",
    "PerformanceEvidence",
    "canonical_json_bytes",
    "fusion_validation_from_dict",
    "kernel_resource_from_code_object",
    "performance_from_rounds",
    "sha256_payload",
]
