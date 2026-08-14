# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Current diagnostic-model governance artifact schemas."""

from enum import StrEnum


class DiagnosticArtifactSchema(StrEnum):
    """Canonical diagnostic calibration, corpus, and acceptance identifiers."""

    DIAGNOSTIC_ACCEPTANCE = "sol_execbench.diagnostic_acceptance.v9"
    DIAGNOSTIC_CALIBRATION = "sol_execbench.diagnostic_calibration.v8"
    DIAGNOSTIC_CASE_REUSE = "sol_execbench.diagnostic_case_reuse.v2"
    DIAGNOSTIC_CORPUS_DESIGN = "sol_execbench.diagnostic_corpus_design.v1"
    DIAGNOSTIC_CORPUS_QUALIFICATION = (
        "sol_execbench.diagnostic_corpus_qualification.v1"
    )
    DIAGNOSTIC_INFERENCE_PROFILE = (
        "sol_execbench.diagnostic_inference_profile.v10"
    )
    DIAGNOSTIC_SOURCE_TRANSITION = (
        "sol_execbench.diagnostic_source_transition.v1"
    )
    DIAGNOSTIC_VALIDATION_CORPUS = (
        "sol_execbench.diagnostic_validation_corpus.v9"
    )
    DIAGNOSTIC_VRAM_WORKING_SET_POLICY = (
        "sol_execbench.diagnostic_vram_working_set_policy.v1"
    )


class DiagnosticAcceptanceArtifactKind(StrEnum):
    """Artifacts in one diagnostic acceptance family."""

    EXPOSURE = "exposure"
    MANIFEST = "manifest"
    RESULT = "result"


class DiagnosticCaseReuseArtifactKind(StrEnum):
    """Artifacts in one held-out case reuse family."""

    HELD_OUT_FRAGMENT = "held_out_fragment"
    MANIFEST = "manifest"


class DiagnosticCorpusDesignArtifactKind(StrEnum):
    """Design input and structural preflight result variants."""

    DESIGN = "design"
    PREFLIGHT = "preflight"


class DiagnosticQualificationArtifactKind(StrEnum):
    """Artifacts in one diagnostic corpus qualification family."""

    GATE = "gate"
    RECEIPT = "receipt"


class DiagnosticCalibrationArtifactKind(StrEnum):
    """Artifacts in one diagnostic calibration contract family."""

    AUDIT = "audit"
    PROFILE = "profile"


class DiagnosticSourceTransitionArtifactKind(StrEnum):
    """Artifacts in one reviewed source transition family."""

    ATTESTATION = "attestation"
    REBIND_RECEIPT = "rebind_receipt"


__all__ = [
    "DiagnosticAcceptanceArtifactKind",
    "DiagnosticArtifactSchema",
    "DiagnosticCalibrationArtifactKind",
    "DiagnosticCaseReuseArtifactKind",
    "DiagnosticCorpusDesignArtifactKind",
    "DiagnosticQualificationArtifactKind",
    "DiagnosticSourceTransitionArtifactKind",
]
