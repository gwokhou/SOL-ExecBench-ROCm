# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""CPU-safe integration tests for the decision CLI sidecar writer.

Covers the IO and budget-loading path that the pure-function unit tests do
not: ``_write_decision_sidecar`` writing ``<trace>.decision.json`` and
``_load_budget_from_environment`` selecting the architecture-matched budget.
"""

from __future__ import annotations

import json
from pathlib import Path

from sol_execbench.cli.sidecars.decision import (
    DECISION_AUTO,
    DECISION_NONE,
    _write_decision_sidecar,
)
from sol_execbench.core.bench.static_kernel.evidence_models import (
    StaticKernelEvidenceClassification,
    StaticKernelEvidenceReasonCode,
    StaticKernelEvidenceSidecar,
    StaticKernelEvidenceStatus,
    StaticResourceFootprint,
)
from sol_execbench.core.platform.arch_capabilities import (
    load_packaged_arch_capability_budget,
)


def _environment_sidecar(path: Path, archs: list[str]) -> None:
    budgets = [
        {
            "status": "available",
            "architecture": arch,
            "budget": load_packaged_arch_capability_budget(arch).model_dump(
                mode="json"
            ),
        }
        for arch in archs
    ]
    path.write_text(json.dumps({"capability_budgets": budgets}), encoding="utf-8")


def _static_evidence(detected: list[str]) -> StaticKernelEvidenceSidecar:
    return StaticKernelEvidenceSidecar(
        schema_version="sol_execbench.static_kernel_evidence.v2",
        status=StaticKernelEvidenceStatus.COLLECTED,
        reason_code=StaticKernelEvidenceReasonCode.STATIC_EVIDENCE_COLLECTED,
        classification=StaticKernelEvidenceClassification(
            detected_architectures=detected, metadata_present=True
        ),
        footprints=[
            StaticResourceFootprint(
                vgpr_used=250, scratch_bytes=1024, spill_detected=True
            )
        ],
    )


def test_decision_auto_writes_sidecar_with_matched_budget(tmp_path: Path) -> None:
    output = tmp_path / "trace.jsonl"
    output.write_text("{}\n")
    env_path = tmp_path / "trace.jsonl.environment.json"
    # Multi-arch environment: gfx942 must be matched, not gfx1150 (first).
    _environment_sidecar(env_path, ["gfx1150", "gfx942"])

    path = _write_decision_sidecar(
        output,
        DECISION_AUTO,
        _static_evidence(["gfx942"]),
        env_path,
        run_id="r1",
        sol_version="v1.43",
    )

    assert path is not None
    assert path == tmp_path / "trace.jsonl.decision.json"
    decision = json.loads(path.read_text(encoding="utf-8"))
    assert decision["schema_version"] == "sol_execbench.decision.v1"
    assert decision["summary"]["architecture"] == "gfx942"  # matched target
    assert decision["authority"] == "diagnostic"
    assert any(h["bottleneck_class"] == "spill_detected" for h in decision["hints"])


def test_decision_none_writes_nothing(tmp_path: Path) -> None:
    output = tmp_path / "trace.jsonl"
    output.write_text("{}\n")
    assert (
        _write_decision_sidecar(
            output, DECISION_NONE, _static_evidence(["gfx942"]), None
        )
        is None
    )
    assert not (tmp_path / "trace.jsonl.decision.json").exists()


def test_decision_auto_without_footprints_writes_nothing(tmp_path: Path) -> None:
    output = tmp_path / "trace.jsonl"
    output.write_text("{}\n")
    empty = StaticKernelEvidenceSidecar(
        schema_version="sol_execbench.static_kernel_evidence.v2",
        status=StaticKernelEvidenceStatus.COLLECTED,
        reason_code=StaticKernelEvidenceReasonCode.STATIC_EVIDENCE_COLLECTED,
    )
    assert _write_decision_sidecar(output, DECISION_AUTO, empty, None) is None
    assert not (tmp_path / "trace.jsonl.decision.json").exists()


def test_decision_auto_without_environment_budget_falls_back_to_partial(
    tmp_path: Path,
) -> None:
    output = tmp_path / "trace.jsonl"
    output.write_text("{}\n")
    # No environment sidecar -> budget None -> PARTIAL with spill-only hints.
    path = _write_decision_sidecar(
        output, DECISION_AUTO, _static_evidence(["gfx942"]), None
    )
    assert path is not None
    decision = json.loads(path.read_text(encoding="utf-8"))
    assert decision["status"] == "partial"
    assert decision["summary"]["architecture"] is None
    assert any(h["bottleneck_class"] == "spill_detected" for h in decision["hints"])


def test_runtime_profile_demotes_pressure_and_keeps_spill(tmp_path: Path) -> None:
    output = tmp_path / "trace.jsonl"
    output.write_text("{}\n")
    env_path = tmp_path / "trace.jsonl.environment.json"
    _environment_sidecar(env_path, ["gfx942"])

    path = _write_decision_sidecar(
        output,
        DECISION_AUTO,
        _static_evidence(["gfx942"]),
        env_path,
        runtime_profile_available=True,
        run_id="r1",
        sol_version="v1.43",
    )
    assert path is not None
    decision = json.loads(path.read_text(encoding="utf-8"))
    # Inferred register pressure is demoted to inferred_low under runtime
    # precedence; deterministic spill stays at inferred_high.
    reg = next(
        h
        for h in decision["hints"]
        if h["bottleneck_class"] == "register_pressure_high"
    )
    assert reg["confidence"] == "inferred_low"
    spill = next(
        h for h in decision["hints"] if h["bottleneck_class"] == "spill_detected"
    )
    assert spill["confidence"] == "inferred_high"
    assert any(
        "Runtime profile takes precedence" in lim for lim in decision["limitations"]
    )
