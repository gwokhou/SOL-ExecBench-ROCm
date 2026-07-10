# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""CPU-safe tests for the static resource footprint model and parser."""

from __future__ import annotations

import json
from pathlib import Path

from sol_execbench.core.bench.static_kernel.evidence_models import (
    StaticKernelEvidenceReasonCode,
    StaticKernelEvidenceStatus,
    StaticKernelEvidenceSidecar,
    StaticResourceFootprint,
)
from sol_execbench.core.bench.static_kernel.footprint_parsers import (
    parse_roc_objdump_resource_usage,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
FIXTURE_DIR = REPO_ROOT / "tests/sol_execbench/fixtures/static_kernel"


def test_parser_extracts_resource_markers():
    text = (
        "; NumVgprs: 40\n; NumSgprs: 20\n; ScratchSize: 1024\n"
        "; LDSByteSize: 512\n; Occupancy: 6\n"
    )
    footprint = parse_roc_objdump_resource_usage(text, artifact_id="k0")

    assert footprint is not None
    assert footprint.vgpr_used == 40
    assert footprint.sgpr_used == 20
    assert footprint.scratch_bytes == 1024
    assert footprint.spill_detected is True
    assert footprint.lds_bytes == 512
    assert footprint.occupancy_estimate_waves_per_cu == 6
    assert footprint.source_tool == "roc-objdump"


def test_parser_returns_none_without_markers():
    assert parse_roc_objdump_resource_usage("no markers here", artifact_id="k0") is None


def test_parser_spill_false_when_zero_scratch():
    footprint = parse_roc_objdump_resource_usage(
        "; NumVgprs: 8\n; ScratchSize: 0\n", artifact_id="k0"
    )
    assert footprint is not None
    assert footprint.spill_detected is False


def test_footprint_governance_flags_diagnostic_only():
    footprint = StaticResourceFootprint(vgpr_used=8)
    assert footprint.diagnostic_only is True
    for flag in (
        "correctness_authority",
        "performance_authority",
        "timing_authority",
        "score_authority",
        "paper_parity_authority",
        "leaderboard_authority",
    ):
        assert getattr(footprint, flag) is False


def test_footprint_round_trip():
    footprint = StaticResourceFootprint(
        vgpr_used=32, sgpr_used=16, spill_detected=False, source_tool="roc-objdump"
    )
    payload = footprint.model_dump(mode="json")
    rebuilt = StaticResourceFootprint.model_validate(payload)
    assert rebuilt.vgpr_used == 32
    assert rebuilt.sgpr_used == 16


def test_valid_footprint_fixture_loads_as_v2_sidecar():
    payload = json.loads(
        (FIXTURE_DIR / "valid.static-footprint.json").read_text(encoding="utf-8")
    )
    sidecar = StaticKernelEvidenceSidecar.model_validate(payload)

    assert sidecar.schema_version == "sol_execbench.static_kernel_evidence.v2"
    assert len(sidecar.footprints) == 1
    footprint = sidecar.footprints[0]
    assert footprint.vgpr_used == 32
    assert footprint.sgpr_used == 16
    assert footprint.spill_detected is False
    assert footprint.diagnostic_only is True
    assert footprint.score_authority is False
    assert footprint.identity is not None
    assert footprint.identity.extractor_tool_id == "roc-objdump"


def test_extractor_collects_footprint_without_downgrading_base_status(tmp_path):
    from sol_execbench.core.bench.static_kernel.evidence_models import (
        StaticKernelEvidenceArtifact,
    )
    from sol_execbench.core.bench.static_kernel.extractors import (
        run_static_kernel_extractors,
    )
    from sol_execbench.core.platform.environment import ProbeCompletedProcess

    shared_object = tmp_path / "k.so"
    shared_object.write_bytes(b"\x7fELF dummy")
    artifact = StaticKernelEvidenceArtifact(
        artifact_id="k0",
        artifact_type="code_object",
        status=StaticKernelEvidenceStatus.COLLECTED,
        reason_code=StaticKernelEvidenceReasonCode.STATIC_EVIDENCE_COLLECTED,
        persisted_path=str(shared_object),
    )
    resource = (
        "; NumVgprs: 40\n; NumSgprs: 20\n; ScratchSize: 1024\n"
        "; LDSByteSize: 512\n; Occupancy: 6\n"
    )

    def probe_runner(command, timeout_seconds):
        return ProbeCompletedProcess(
            returncode=0, stdout=f"{command[0]} version", stderr=""
        )

    def extractor_runner(command, timeout_seconds):
        if command[0] == "roc-objdump":
            return ProbeCompletedProcess(returncode=0, stdout=resource, stderr="")
        return ProbeCompletedProcess(returncode=0, stdout="disassembly", stderr="")

    sidecar = run_static_kernel_extractors(
        artifacts=[artifact],
        evidence_directory=tmp_path,
        sidecar_base_directory=tmp_path,
        runner=extractor_runner,
        probe_runner=probe_runner,
        which=lambda binary: f"/usr/bin/{binary}",
    )

    assert sidecar.status.value == "collected"
    assert {run.tool_id for run in sidecar.tool_runs} == {"llvm-objdump", "readelf"}
    assert len(sidecar.footprints) == 1
    footprint = sidecar.footprints[0]
    assert footprint.vgpr_used == 40
    assert footprint.scratch_bytes == 1024
    assert footprint.spill_detected is True
    assert footprint.occupancy_estimate_waves_per_cu == 6


def test_extractor_skips_footprint_when_roc_objdump_missing(tmp_path):
    from sol_execbench.core.bench.static_kernel.evidence_models import (
        StaticKernelEvidenceArtifact,
    )
    from sol_execbench.core.bench.static_kernel.extractors import (
        run_static_kernel_extractors,
    )
    from sol_execbench.core.platform.environment import ProbeCompletedProcess

    shared_object = tmp_path / "k.so"
    shared_object.write_bytes(b"\x7fELF dummy")
    artifact = StaticKernelEvidenceArtifact(
        artifact_id="k0",
        artifact_type="code_object",
        status=StaticKernelEvidenceStatus.COLLECTED,
        reason_code=StaticKernelEvidenceReasonCode.STATIC_EVIDENCE_COLLECTED,
        persisted_path=str(shared_object),
    )

    def which(binary):
        return None if binary == "roc-objdump" else f"/usr/bin/{binary}"

    def probe_runner(command, timeout_seconds):
        return ProbeCompletedProcess(returncode=0, stdout="v", stderr="")

    def extractor_runner(command, timeout_seconds):
        return ProbeCompletedProcess(returncode=0, stdout="disassembly", stderr="")

    sidecar = run_static_kernel_extractors(
        artifacts=[artifact],
        evidence_directory=tmp_path,
        sidecar_base_directory=tmp_path,
        runner=extractor_runner,
        probe_runner=probe_runner,
        which=which,
    )

    assert sidecar.status.value == "collected"
    assert sidecar.footprints == []


def test_parser_does_not_match_total_marker_substring():
    # NumVgprs is a substring of TotalNumVgprs; the anchored regex must read
    # only the plain NumVgprs line, not the Total line. SGPR real-world output
    # is TotalNumSgprs, matched via the (?:Total)? optional prefix.
    text = (
        ";  NumVgprs: 40\n;  TotalNumVgprs: 256\n"
        ";  TotalNumSgprs: 80\n;  ScratchSize: 0\n"
    )
    footprint = parse_roc_objdump_resource_usage(text, artifact_id="k0")
    assert footprint is not None
    assert footprint.vgpr_used == 40  # not 256 from TotalNumVgprs
    assert footprint.sgpr_used == 80  # TotalNumSgprs matched via (?:Total)?
