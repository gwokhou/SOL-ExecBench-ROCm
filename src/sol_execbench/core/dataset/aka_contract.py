# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Shared symbolic values for the AKA corpus contract."""

from enum import StrEnum


class AkaCorpusRole(StrEnum):
    """Scoring treatment for one authored AKA problem."""

    SCORED = "scored"
    COMPATIBILITY_SENTINEL = "compatibility_sentinel"
    TARGET_INCOMPATIBLE = "target_incompatible"


class AkaArtifactRole(StrEnum):
    """Semantic role of one content-addressed upstream AKA file."""

    CONFIG = "config"
    SEMANTIC_REFERENCE = "semantic_reference"
    CORRECTNESS_RUNNER = "correctness_runner"


class AkaSuite(StrEnum):
    """AKA suites admitted by the corpus conversion policy."""

    TORCH2HIP = "torch2hip"
    TORCH2FLYDSL = "torch2flydsl"
    INSTRUCTION2TRITON = "instruction2triton"


class AkaPassKind(StrEnum):
    """Execution direction represented by an authored problem."""

    FORWARD = "forward"
    BACKWARD = "backward"


class AkaFusionDepth(StrEnum):
    """Whether a problem is a primitive or fused computation."""

    SINGLE = "single"
    FUSED = "fused"


class AkaSourceFamily(StrEnum):
    """Upstream AKA source family represented in this corpus."""

    KERNELBENCH = "kernelbench"
    GPUMODE = "gpumode"
    ROCMBENCH = "rocmbench"
    FLYDSL = "flydsl"


class AkaOperation(StrEnum):
    """Operation strata used by the AKA corpus coverage policy."""

    MATMUL = "matmul"
    SOFTMAX = "softmax"
    ELEMENTWISE = "elementwise"
    NORM = "norm"
    CONV = "conv"
    ATTENTION = "attention"


class AkaOfficialScoringStatus(StrEnum):
    """Publication state of the official AKA scoring contract."""

    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"


class AkaRequiredEvidenceKind(StrEnum):
    """Content-addressed evidence classes required for an official score."""

    RELEASE_BASELINE = "content_addressed_release_baseline"
    RELEASE_CANDIDATE = "content_addressed_candidate_execution"
    SOLAR_MANIFESTS = "pinned_solar_manifests"


class AkaReleasePolicy(StrEnum):
    """Closed publication policies admitted by the AKA scorer."""

    CONTENT_ADDRESSED_PUBLISHER_V1 = "content_addressed_publisher_v1"


AKA_MANIFEST_SCHEMA_VERSION = 5
AKA_TOLERANCE_CALIBRATION_FILENAME = "tolerance-calibration.json"
AKA_OFFICIAL_BASELINE_ID = "rx9060xt-gfx1200-reference-v1"
AKA_REQUIRED_RELEASE_EVIDENCE = tuple(AkaRequiredEvidenceKind)

__all__ = [
    "AKA_MANIFEST_SCHEMA_VERSION",
    "AKA_OFFICIAL_BASELINE_ID",
    "AKA_REQUIRED_RELEASE_EVIDENCE",
    "AKA_TOLERANCE_CALIBRATION_FILENAME",
    "AkaArtifactRole",
    "AkaCorpusRole",
    "AkaFusionDepth",
    "AkaOfficialScoringStatus",
    "AkaOperation",
    "AkaPassKind",
    "AkaRequiredEvidenceKind",
    "AkaReleasePolicy",
    "AkaSourceFamily",
    "AkaSuite",
]
