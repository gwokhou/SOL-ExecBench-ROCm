# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Compatibility facade for CLI diagnostic sidecar helpers."""

from __future__ import annotations

from .agent_feedback import (
    _agent_feedback_artifact_citations as _agent_feedback_artifact_citations,
    _agent_feedback_identity_fields as _agent_feedback_identity_fields,
    _agent_feedback_run_id as _agent_feedback_run_id,
    _agent_feedback_sidecar_path as _agent_feedback_sidecar_path,
    _write_agent_feedback_sidecar as _write_agent_feedback_sidecar,
)
from .profile import (
    _profile_output_directory as _profile_output_directory,
    _profile_sidecar_path as _profile_sidecar_path,
    _profile_summary_artifact_citations as _profile_summary_artifact_citations,
    _profile_summary_sidecar_path as _profile_summary_sidecar_path,
    _write_profile_sidecar as _write_profile_sidecar,
    _write_profile_summary_sidecar as _write_profile_summary_sidecar,
)
from .static_evidence import (
    STATIC_EVIDENCE_AUTO as STATIC_EVIDENCE_AUTO,
    STATIC_EVIDENCE_NONE as STATIC_EVIDENCE_NONE,
    _collect_static_evidence_for_cli as _collect_static_evidence_for_cli,
    _static_evidence_directory as _static_evidence_directory,
    _static_evidence_payload as _static_evidence_payload,
    _static_evidence_sidecar_path as _static_evidence_sidecar_path,
    _static_evidence_summary as _static_evidence_summary,
    _write_static_evidence_sidecar as _write_static_evidence_sidecar,
)
