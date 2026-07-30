# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Shared symbolic values for the AKA corpus contract."""

from enum import StrEnum


class AKACorpusRole(StrEnum):
    """Scoring treatment for one authored AKA problem."""

    SCORED = "scored"
    COMPATIBILITY_SENTINEL = "compatibility_sentinel"
    TARGET_INCOMPATIBLE = "target_incompatible"


class AKAArtifactRole(StrEnum):
    """Semantic role of one content-addressed upstream AKA file."""

    CONFIG = "config"
    SEMANTIC_REFERENCE = "semantic_reference"
    CORRECTNESS_RUNNER = "correctness_runner"


class AKASuite(StrEnum):
    """AKA suites admitted by the corpus conversion policy."""

    TORCH2HIP = "torch2hip"
    TORCH2FLYDSL = "torch2flydsl"
    INSTRUCTION2TRITON = "instruction2triton"


class AKAPassKind(StrEnum):
    """Execution direction represented by an authored problem."""

    FORWARD = "forward"
    BACKWARD = "backward"


class AKAFusionDepth(StrEnum):
    """Whether a problem is a primitive or fused computation."""

    SINGLE = "single"
    FUSED = "fused"


class AKASourceFamily(StrEnum):
    """Upstream AKA source family represented in this corpus."""

    KERNELBENCH = "kernelbench"
    GPUMODE = "gpumode"
    ROCMBENCH = "rocmbench"
    FLYDSL = "flydsl"


class AKAOperation(StrEnum):
    """Operation strata used by the AKA corpus coverage policy."""

    MATMUL = "matmul"
    SOFTMAX = "softmax"
    ELEMENTWISE = "elementwise"
    NORM = "norm"
    CONV = "conv"
    ATTENTION = "attention"
    LOSS = "loss"
    QUANTIZATION = "quantization"
    ROUTING = "routing"
    POSITION_ENCODING = "position_encoding"


class AKACapability(StrEnum):
    """Mechanically enforced corpus expansion capabilities."""

    BOUNDED_INTEGER_INPUT = "bounded_integer_input"
    POSITIVE_INPUT = "positive_input"
    SIMPLEX_INPUT = "simplex_input"
    SCALAR_TENSOR_OUTPUT = "scalar_tensor_output"
    MULTI_OUTPUT = "multi_output"
    MIXED_OUTPUT_DTYPE = "mixed_output_dtype"
    CODE_DISTANCE = "code_distance"
    PARTIAL_CUSTOM_INPUT = "partial_custom_input"
    STRUCTURED_OFFSETS = "structured_offsets"
    FP8_OUTPUT = "fp8_output"
    UINT8_OUTPUT = "uint8_output"
    RAW_CODE_DISTANCE = "raw_code_distance"
    COUPLED_TOPK = "coupled_topk"


class AKAOfficialScoringStatus(StrEnum):
    """Publication state of the official AKA scoring contract."""

    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"


class AKARequiredEvidenceKind(StrEnum):
    """Content-addressed evidence classes required for an official score."""

    RELEASE_BASELINE = "content_addressed_release_baseline"
    RELEASE_CANDIDATE = "content_addressed_candidate_execution"
    SOLAR_MANIFESTS = "pinned_solar_manifests"


class AKAReleasePolicy(StrEnum):
    """Closed publication policies admitted by the AKA scorer."""

    CONTENT_ADDRESSED_PUBLISHER_V1 = "content_addressed_publisher_v1"


class AKACompatibilityStage(StrEnum):
    """Stages that may include or exclude one AKA workload."""

    STATIC = "static"
    LIVE_PROBE = "live_probe"


class AKAProbeStatus(StrEnum):
    """Closed worker response states for a live compatibility probe."""

    COMPATIBLE = "compatible"
    INCOMPATIBLE = "incompatible"
    INFRASTRUCTURE_ERROR = "infrastructure_error"


class AKATargetGeneration(StrEnum):
    """GPU generations admitted by the authored AKA target catalog."""

    CDNA3 = "cdna3"
    RDNA3_5 = "rdna3_5"
    RDNA4 = "rdna4"


AKA_TOLERANCE_CALIBRATION_FILENAME = "tolerance-calibration.json"
AKA_OFFICIAL_BASELINE_ID = "rx9060xt-gfx1200-reference-v2"
AKA_REQUIRED_RELEASE_EVIDENCE = tuple(AKARequiredEvidenceKind)

__all__ = [
    "AKA_OFFICIAL_BASELINE_ID",
    "AKA_REQUIRED_RELEASE_EVIDENCE",
    "AKA_TOLERANCE_CALIBRATION_FILENAME",
    "AKAArtifactRole",
    "AKACapability",
    "AKACompatibilityStage",
    "AKACorpusRole",
    "AKAFusionDepth",
    "AKAOfficialScoringStatus",
    "AKAOperation",
    "AKAPassKind",
    "AKAProbeStatus",
    "AKAReleasePolicy",
    "AKARequiredEvidenceKind",
    "AKASourceFamily",
    "AKASuite",
    "AKATargetGeneration",
]
