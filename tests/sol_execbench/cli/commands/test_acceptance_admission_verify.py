# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
"""Admission-side verification of the acceptance result (audit Gap 1/2/3).

When the source manifest is supplied to ``agent-feedback``, the admission
re-derives the verdict from the manifest and rejects drift, a forged
``manifest_sha256``, and a vacuous action policy.
"""

from __future__ import annotations

import pytest

from sol_execbench.cli.commands.diagnostics import (
    _verify_acceptance_against_manifest,
)
from sol_execbench.core.bench.performance_model.acceptance import (
    DiagnosticAcceptanceCase,
    DiagnosticAcceptanceManifest,
    evaluate_diagnostic_acceptance,
)
from sol_execbench.core.bench.performance_model.models import (
    CalibrationIdentity,
    DiagnosticModelIdentity,
    WorkloadKind,
)
from sol_execbench.core.integrity import stable_json_checksum

_ACTIONS = {
    WorkloadKind.ELEMENTWISE: "stop_launch_bound_search",
    WorkloadKind.TRANSPOSE: "improve_coalescing",
    WorkloadKind.REDUCTION: "reduce_lds_barriers",
    WorkloadKind.MATMUL: "restore_wmma_path",
    WorkloadKind.SOFTMAX: "stop_launch_bound_search",
    WorkloadKind.CROSS_ENTROPY: "improve_coalescing",
    WorkloadKind.INDEXED_READ: "improve_coalescing",
    WorkloadKind.INDEXED_UPDATE: "reduce_atomic_contention",
    WorkloadKind.COMPOSITE: "reduce_dispatch_count",
    WorkloadKind.TRANSFORMER: "restore_fused_attention_path",
    WorkloadKind.CONCURRENT: "remove_extra_traffic",
}


def _identity() -> CalibrationIdentity:
    return CalibrationIdentity(
        gpu_architecture="gfx1200",
        gpu_id="gpu-0",
        gpu_bdf="0000:03:00.0",
        rocm_version="7.2",
        compiler_version="hipcc-7.2",
        clock_mode="locked",
        power_profile="stable_peak",
    )


def _model_identity() -> DiagnosticModelIdentity:
    return DiagnosticModelIdentity(
        model_version="gfx1200_diagnostic.v6",
        policy_files={"policy.py": "d" * 64},
        counter_semantics_sha256="e" * 64,
        policy_bundle_sha256="f" * 64,
    )


def _case(kind: WorkloadKind, index: int) -> DiagnosticAcceptanceCase:
    identity = f"{kind}:{index}"
    action = _ACTIONS[kind]
    return DiagnosticAcceptanceCase(
        case_id=identity,
        pair_id=stable_json_checksum([identity, "pair"]),
        workload_kind=kind,
        evidence_manifest_sha256=stable_json_checksum([identity, "evidence"]),
        performance_diagnostic_sha256=stable_json_checksum(
            [identity, "diagnostic"]
        ),
        predicted_ms=1.05,
        lower_ms=0.9,
        upper_ms=1.1,
        measured_ms=1.0,
        predicted_action_codes=[action],
        gold_action_codes=[action],
    )


def _manifest(*, vacuous: bool = False) -> DiagnosticAcceptanceManifest:
    return DiagnosticAcceptanceManifest(
        model_identity=_model_identity(),
        calibration_profile_sha256="a" * 64,
        calibration_identity=_identity(),
        inference_profile_sha256="b" * 64,
        development_corpus_sha256="c" * 64,
        held_out_corpus_sha256="d" * 64,
        enabled_action_codes=[] if vacuous else sorted(_ACTIONS.values()),
        cases=[_case(kind, index) for kind in _ACTIONS for index in range(20)],
    )


def test_valid_manifest_passes_verification() -> None:
    manifest = _manifest()
    result = evaluate_diagnostic_acceptance(manifest)

    _verify_acceptance_against_manifest(result, manifest)  # must not raise


def test_forged_manifest_hash_rejected() -> None:
    """Gap 2: a self-stamped manifest_sha256 that no longer matches is rejected."""
    manifest = _manifest()
    result = evaluate_diagnostic_acceptance(manifest)
    forged = result.model_copy(update={"manifest_sha256": "0" * 64})

    with pytest.raises(ValueError, match="manifest hash does not match"):
        _verify_acceptance_against_manifest(forged, manifest)


def test_forged_metric_rejected() -> None:
    """Gap 1: a result field that disagrees with the cited manifest is rejected."""
    manifest = _manifest()
    result = evaluate_diagnostic_acceptance(manifest)
    forged = result.model_copy(update={"median_absolute_percentage_error": 5.0})

    with pytest.raises(ValueError, match="disagrees with the cited manifest"):
        _verify_acceptance_against_manifest(forged, manifest)


def test_vacuous_action_policy_rejected() -> None:
    """Gap 3: an accepted manifest that enables no action is rejected."""
    manifest = _manifest(vacuous=True)
    result = evaluate_diagnostic_acceptance(manifest)
    assert (
        result.accepted is True
    )  # current evaluator still admits it vacuously

    with pytest.raises(ValueError, match="vacuous action policy"):
        _verify_acceptance_against_manifest(result, manifest)
