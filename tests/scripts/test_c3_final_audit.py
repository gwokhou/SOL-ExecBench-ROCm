"""Unit tests for the post-C3 device-free audit and full-matrix summarizer."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.internal.solar import c2_c3_campaign, c3_final_audit
from sol_execbench.core.integrity import sha256_file
from sol_execbench.core.solar_bridge.coverage_models import (
    SLOT_ORDER,
    CoverageAttempt,
    CoverageOptions,
    CoverageRow,
    CoverageStatus,
    TargetCoverageReport,
)
from solar.ir.contracts import IRPath

_FAKE_HASH = "a" * 64


def _slot_role(slot: str) -> str:
    if slot == "smoke":
        return "smoke"
    return "development" if slot.endswith("-low") else "holdout"


def _make_row(
    *,
    semantic_id: str,
    slot: str,
    ir_path: IRPath,
    family: str,
    status: CoverageStatus,
    reason: str,
    attempt: CoverageAttempt,
) -> CoverageRow:
    uuid = f"00000000-0000-5000-8000-{semantic_id[-4:].encode().hex()}{SLOT_ORDER.index(slot):04d}"[
        :36
    ]
    return CoverageRow(
        key=f"M0-C0-20261107-r1/{_FAKE_HASH[:8]}/{semantic_id}/{uuid}/{ir_path.value}",
        semantic_id=semantic_id,
        workload_uuid=uuid,
        ir_path=ir_path,
        slot=slot,
        role=_slot_role(slot),
        family=family,
        policy={"tolerance": "fp16"},
        attempts=[attempt],
        status=status,
        reason=reason,
    )


def _passed_attempt(
    *, resource_work: dict[str, object], flops: int = 0
) -> CoverageAttempt:
    return CoverageAttempt(
        status=CoverageStatus.PASSED,
        wall_seconds=1.5,
        readiness={
            "status": "ready",
            "stages": [
                {"stage": "graph_extraction", "status": "passed"},
                {"stage": "ir_conversion", "status": "passed"},
                {"stage": "conversion_verification", "status": "passed"},
            ],
        },
        formal={
            "stage": None,
            "reason_code": None,
            "bound_kind": "tile_v1",
            "limiting_resource": "valu",
            "lower_bound_seconds": 1e-6,
        },
        observations={
            "total": {
                "flops": flops,
                "macs": flops // 2,
                "resource_work": resource_work,
                "lower_bound_seconds": 1e-6,
            },
            "metadata": {
                "external_reads": {"quality": "exact", "bytes": 65536},
                "views": {
                    "unfused": {"status": "available", "reason": "ready"},
                    "ideal_fused": {"status": "available", "reason": "ready"},
                    "cache_aware": {"status": "available", "reason": "ready"},
                },
            },
            "verification": {
                "numerical": {"status": "passed", "cases": []},
                "gradient": [{"status": "passed", "inputs_verified": 3.0}],
                "effects": {"status": "not_observable", "reason": "none"},
            },
            "readiness": {
                "records": [{"stage": "graph_extraction", "wall_seconds": 0.4}]
            },
            "formal": {
                "records": [{"stage": "formal_analysis", "wall_seconds": 0.2}]
            },
        },
    )


def _failed_attempt(
    *,
    stage: str,
    reason_code: str,
    message: str,
    formal: bool,
) -> CoverageAttempt:
    readiness = (
        {
            "status": "ready",
            "stages": [
                {"stage": "conversion_verification", "status": "passed"}
            ],
        }
        if formal
        else {
            "status": "failed",
            "failure_stage": stage,
            "reason_code": reason_code,
            "message": message,
            "stages": [{"stage": stage, "status": "failed"}],
        }
    )
    formal_dict = (
        {
            "stage": stage,
            "reason_code": reason_code,
            "message": message,
        }
        if formal
        else None
    )
    return CoverageAttempt(
        status=CoverageStatus.FAILED,
        wall_seconds=2.5,
        reason=reason_code,
        readiness=readiness,
        formal=formal_dict,
        observations={
            "readiness": {"records": [{"stage": stage, "wall_seconds": 0.5}]}
        },
    )


def test_two_path_comparison_keys_by_slot_and_tracks_divergence(
    tmp_path: Path,
) -> None:
    rows: list[CoverageRow] = []
    for slot in SLOT_ORDER:
        rows.append(
            _make_row(
                semantic_id="llm.norm_activation.rmsnorm.v2",
                slot=slot,
                ir_path=IRPath.MAKE_FX_ATEN,
                family="norm_activation",
                status=CoverageStatus.PASSED,
                reason="completed",
                attempt=_passed_attempt(resource_work={"sfu": {"fp32": 100}}),
            )
        )
        rows.append(
            _make_row(
                semantic_id="llm.norm_activation.rmsnorm.v2",
                slot=slot,
                ir_path=IRPath.TORCHVIEW_EXTENDED_EINSUM,
                family="norm_activation",
                status=CoverageStatus.PASSED,
                reason="completed",
                attempt=_passed_attempt(resource_work={"valu": {"fp32": 100}}),
            )
        )
        mfx_status = (
            CoverageStatus.FAILED
            if slot == "capacity-high"
            else CoverageStatus.PASSED
        )
        mfx_attempt = (
            _failed_attempt(
                stage="formal_analysis",
                reason_code="formal_analysis_failed",
                message="strict analysis did not produce a complete tile-aware bound",
                formal=True,
            )
            if slot == "capacity-high"
            else _passed_attempt(resource_work={"mfma": {"fp16": 64}}, flops=64)
        )
        rows.append(
            _make_row(
                semantic_id="llm.linear.matmul_bias.v2",
                slot=slot,
                ir_path=IRPath.MAKE_FX_ATEN,
                family="linear",
                status=mfx_status,
                reason="formal_analysis_failed"
                if slot == "capacity-high"
                else "completed",
                attempt=mfx_attempt,
            )
        )
        rows.append(
            _make_row(
                semantic_id="llm.linear.matmul_bias.v2",
                slot=slot,
                ir_path=IRPath.TORCHVIEW_EXTENDED_EINSUM,
                family="linear",
                status=CoverageStatus.FAILED,
                reason="formal_analysis_failed",
                attempt=_failed_attempt(
                    stage="formal_analysis",
                    reason_code="formal_analysis_failed",
                    message="strict analysis did not produce a complete tile-aware bound",
                    formal=True,
                ),
            )
        )

    report = TargetCoverageReport(
        identity={"synthetic": True},
        identity_sha256=_FAKE_HASH,
        decisions=[],
        rows=rows,
        elapsed_seconds=60.0,
        smoke_elapsed_seconds=8.0,
        matrix_complete=True,
        costs_observed=True,
        complete=True,
    )
    coverage_dir = tmp_path / "coverage"
    coverage_dir.mkdir()
    paths = c2_c3_campaign.CampaignPaths(
        repo=tmp_path, outputs=tmp_path, freeze=tmp_path / "freeze.json"
    )
    summary = c3_final_audit.build_c3_summary(
        paths, report, CoverageOptions(orojenesis_home=tmp_path)
    )
    two_path = summary["two_path_comparison"]
    assert two_path["total_workload_pairs"] == 18
    assert two_path["common_passed_pair_count"] == 9
    assert two_path["common_passed_pairs"][0][
        "differing_resource_work_keys"
    ] == ["sfu", "valu"]
    consistency = two_path["by_definition_consistency"]
    assert (
        consistency["llm.norm_activation.rmsnorm.v2"]["all_common_flops_equal"]
        is True
    )
    assert (
        consistency["llm.norm_activation.rmsnorm.v2"][
            "all_common_resource_work_equal"
        ]
        is False
    )
    matmul = consistency["llm.linear.matmul_bias.v2"]
    assert matmul["make_fx_passed_slots_count"] == 8
    assert matmul["slot_status_uniform_make_fx"] is False
    assert matmul["slot_status_uniform_torchview"] is True

    ledger = c3_final_audit.build_c3_exposure_ledger(
        c2_summary={
            "target_view_manifest_sha256": _FAKE_HASH,
            "workload_view_digest": _FAKE_HASH,
        },
        report=report,
        freeze_sha256=_FAKE_HASH,
    )
    assert ledger["exposed_rows_count"] == 36
    assert ledger["exposed_rows_by_role"] == {
        "development": 16,
        "holdout": 16,
        "smoke": 4,
    }


def test_m0_classification_and_definition_deduplicated_gap_ranking() -> None:
    rows = [
        _make_row(
            semantic_id="llm.attention.mqa_decode.v2",
            slot="smoke",
            ir_path=IRPath.TORCHVIEW_EXTENDED_EINSUM,
            family="attention",
            status=CoverageStatus.FAILED,
            reason="strict_conversion_failed",
            attempt=_failed_attempt(
                stage="ir_conversion",
                reason_code="strict_conversion_failed",
                message="cannot exactly convert Model.expand_as: Unsupported operation: expand_as",
                formal=False,
            ),
        ),
        _make_row(
            semantic_id="llm.attention.sliding_window.v2",
            slot="smoke",
            ir_path=IRPath.TORCHVIEW_EXTENDED_EINSUM,
            family="attention",
            status=CoverageStatus.FAILED,
            reason="exact_replay_failed",
            attempt=_failed_attempt(
                stage="conversion_verification",
                reason_code="exact_replay_failed",
                message="missing external tensor Model.hidden-tensor_8.Output",
                formal=False,
            ),
        ),
        _make_row(
            semantic_id="llm.attention.attention_sink.v2",
            slot="smoke",
            ir_path=IRPath.TORCHVIEW_EXTENDED_EINSUM,
            family="attention",
            status=CoverageStatus.FAILED,
            reason="gradient_verification_failed",
            attempt=_failed_attempt(
                stage="conversion_verification",
                reason_code="gradient_verification_failed",
                message="numerical mismatch: matched_ratio=0",
                formal=False,
            ),
        ),
        _make_row(
            semantic_id="llm.attention.sliding_window.v2",
            slot="smoke",
            ir_path=IRPath.MAKE_FX_ATEN,
            family="attention",
            status=CoverageStatus.FAILED,
            reason="formal_analysis_failed",
            attempt=_failed_attempt(
                stage="formal_analysis",
                reason_code="formal_analysis_failed",
                message="operation 'arange' has no amd_resource_v3 rule",
                formal=True,
            ),
        ),
        *[
            _make_row(
                semantic_id="llm.linear.single_def_nine_slots.v2",
                slot=slot,
                ir_path=IRPath.TORCHVIEW_EXTENDED_EINSUM,
                family="linear",
                status=CoverageStatus.FAILED,
                reason="formal_analysis_failed",
                attempt=_failed_attempt(
                    stage="formal_analysis",
                    reason_code="formal_analysis_failed",
                    message="strict analysis did not produce a complete tile-aware bound",
                    formal=True,
                ),
            )
            for slot in SLOT_ORDER
        ],
        *[
            _make_row(
                semantic_id=sem,
                slot="smoke",
                ir_path=ir_path,
                family="position",
                status=CoverageStatus.FAILED,
                reason="strict_conversion_failed",
                attempt=_failed_attempt(
                    stage="ir_conversion",
                    reason_code="strict_conversion_failed",
                    message="cannot exactly convert x: Unsupported operation: shared_gap",
                    formal=False,
                ),
            )
            for sem in ("llm.position.rope_a.v2", "llm.position.rope_b.v2")
            for ir_path in (
                IRPath.MAKE_FX_ATEN,
                IRPath.TORCHVIEW_EXTENDED_EINSUM,
            )
        ],
    ]
    cats = [c3_final_audit.classify_m0_reporting(r) for r in rows[:4]]
    assert cats == [
        ("unsupported_or_unproved", "ir_unsupported_op"),
        ("failed", "exact_replay_failed"),
        ("failed", "gradient_verification_failed"),
        ("unsupported_or_unproved", "missing_resource_rule"),
    ]
    gaps = c3_final_audit.build_gap_inventory(rows)
    assert gaps[0]["gap_id"] == "ir_conversion:unsupported_op:shared_gap"
    assert gaps[0]["affected_definitions_count"] == 2
    assert gaps[0]["affected_rows_count"] == 4
    assert gaps[0]["l3d_frontend_candidate"] is True
    tile_gap = next(
        g
        for g in gaps
        if g["gap_id"] == "formal_analysis:incomplete_tile_aware_bound"
    )
    assert tile_gap["affected_definitions_count"] == 1
    assert tile_gap["affected_rows_count"] == 9
    assert tile_gap["l3d_frontend_candidate"] is False


def test_verify_preserved_artifacts_rejects_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert set(c3_final_audit.PRESERVED_SMOKE_SHA256) == {
        "llm-core/capacity-evidence.json",
        "llm-core/c2-summary.json",
        "llm-core/smoke-summary.json",
        "llm-core/exposure-ledger.json",
        "llm-core/cpu-audit.json",
    }
    target = tmp_path / "llm-core/smoke-summary.json"
    target.parent.mkdir(parents=True)
    target.write_text('{"ok": true}\n', encoding="utf-8")
    expected_sha = sha256_file(target)
    monkeypatch.setattr(
        c3_final_audit,
        "PRESERVED_SMOKE_SHA256",
        {"llm-core/smoke-summary.json": expected_sha},
    )
    assert c3_final_audit.verify_preserved_artifacts(tmp_path) == {
        "llm-core/smoke-summary.json": expected_sha
    }
    target.write_text('{"ok": false}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="preserved artifact modified"):
        c3_final_audit.verify_preserved_artifacts(tmp_path)


def test_budget_stop_and_r2_failure_classifications(tmp_path: Path) -> None:
    oom_extract = _failed_attempt(
        stage="graph_extraction",
        reason_code="graph_extraction_failed",
        message="device out of memory. Tried to allocate 5.68 GiB.",
        formal=False,
    )
    num_equiv = _failed_attempt(
        stage="conversion_verification",
        reason_code="numerical_equivalence_failed",
        message="numerical mismatch: matched_ratio=0.999",
        formal=False,
    )
    mapper_to = _failed_attempt(
        stage="formal_analysis",
        reason_code="toolchain_unavailable",
        message="Orojenesis execution failed",
        formal=True,
    )
    mapper_to.wall_seconds = 7230.0
    rows = [
        _make_row(
            semantic_id="llm.attention.gqa_causal.v2",
            slot="capacity-high",
            ir_path=IRPath.TORCHVIEW_EXTENDED_EINSUM,
            family="attention",
            status=CoverageStatus.FAILED,
            reason="graph_extraction_failed",
            attempt=oom_extract,
        ),
        _make_row(
            semantic_id="llm.attention.gqa_causal.v2",
            slot="capacity-high",
            ir_path=IRPath.MAKE_FX_ATEN,
            family="attention",
            status=CoverageStatus.UNPROVED,
            reason="conversion_not_proven",
            attempt=_failed_attempt(
                stage="conversion_verification",
                reason_code="conversion_not_proven",
                message="device out of memory during verification replay",
                formal=False,
            ),
        ),
        _make_row(
            semantic_id="llm.attention.attention_sink.v2",
            slot="irregular-low",
            ir_path=IRPath.TORCHVIEW_EXTENDED_EINSUM,
            family="attention",
            status=CoverageStatus.FAILED,
            reason="numerical_equivalence_failed",
            attempt=num_equiv,
        ),
        _make_row(
            semantic_id="llm.attention.local_global.v2",
            slot="throughput-high",
            ir_path=IRPath.MAKE_FX_ATEN,
            family="attention",
            status=CoverageStatus.FAILED,
            reason="toolchain_unavailable",
            attempt=mapper_to,
        ),
        CoverageRow(
            key="M0-C0-20261107-r1/aaaaaaaa/llm.linear.grouped_matmul.v2/uuid/make_fx_aten",
            semantic_id="llm.linear.grouped_matmul.v2",
            workload_uuid="00000000-0000-5000-8000-000000000099",
            ir_path=IRPath.MAKE_FX_ATEN,
            slot="latency-low",
            role="development",
            family="linear",
            policy={"tolerance": "fp16"},
            attempts=[],
            status=CoverageStatus.NOT_REACHED,
            reason="activity budget exhausted",
        ),
    ]
    cats = [c3_final_audit.classify_m0_reporting(r) for r in rows]
    assert cats == [
        ("budget_exhausted", "graph_extraction_device_oom"),
        ("unsupported_or_unproved", "conversion_not_proven_device_oom"),
        ("failed", "numerical_equivalence_failed"),
        (
            "budget_exhausted",
            "mapper_timeout_surfaced_as_toolchain_unavailable",
        ),
        ("not_reached", "activity budget exhausted"),
    ]
    gaps = c3_final_audit.build_gap_inventory(rows)
    gap_ids = {g["gap_id"] for g in gaps}
    assert "not_reached" not in str(gap_ids)
    oom_gap = next(
        g
        for g in gaps
        if g["gap_id"] == "graph_extraction:device_oom_eager_trace"
    )
    assert oom_gap["l3d_frontend_candidate"] is False

    budget_stop = c3_final_audit._build_budget_stop_summary(
        rows,
        TargetCoverageReport(
            identity={"synthetic": True},
            identity_sha256=_FAKE_HASH,
            decisions=[],
            rows=rows,
            elapsed_seconds=43200.5,
            smoke_elapsed_seconds=450.0,
            matrix_complete=False,
            costs_observed=False,
            complete=False,
        ),
        CoverageOptions(orojenesis_home=tmp_path),
    )
    assert budget_stop["stopped_by_activity_budget"] is True
    assert budget_stop["not_reached_rows_count"] == 1
    assert budget_stop["first_not_reached_row"]["semantic_id"] == (
        "llm.linear.grouped_matmul.v2"
    )
    assert len(budget_stop["mapper_timeout_toolchain_rows"]) == 1


def test_verify_report_and_artifact_integrity_handles_timeout_transient_drift(
    tmp_path: Path,
) -> None:
    cov = tmp_path / "coverage"
    transient = cov / "row-000/attempt-00/.formal.tmp/oaves.csv"
    readiness = cov / "row-000/attempt-00/readiness/operator_graph.yaml"
    transient.parent.mkdir(parents=True)
    readiness.parent.mkdir(parents=True)
    transient.write_text("before\n", encoding="utf-8")
    readiness.write_text("ready\n", encoding="utf-8")

    attempt_ok = CoverageAttempt(
        status=CoverageStatus.BUDGET_EXHAUSTED,
        wall_seconds=10.0,
        reason="worker_timeout",
        artifacts={
            "row-000/attempt-00/.formal.tmp/oaves.csv": sha256_file(transient),
            "row-000/attempt-00/readiness/operator_graph.yaml": sha256_file(
                readiness
            ),
        },
    )
    row = _make_row(
        semantic_id="llm.linear.batched_matmul.v2",
        slot="capacity-high",
        ir_path=IRPath.TORCHVIEW_EXTENDED_EINSUM,
        family="linear",
        status=CoverageStatus.BUDGET_EXHAUSTED,
        reason="worker_timeout",
        attempt=attempt_ok,
    )
    expected = TargetCoverageReport(
        identity={"synthetic": True},
        identity_sha256=_FAKE_HASH,
        decisions=[],
        rows=[
            row.model_copy(
                update={
                    "attempts": [],
                    "status": CoverageStatus.NOT_REACHED,
                    "reason": "not scheduled",
                }
            )
        ],
    )
    report = TargetCoverageReport(
        identity={"synthetic": True},
        identity_sha256=_FAKE_HASH,
        decisions=[],
        rows=[row],
    )

    transient.write_text("after-orphan-flush\n", encoding="utf-8")
    drifts = c3_final_audit._verify_report_and_artifact_integrity(
        report, expected, cov
    )
    assert len(drifts) == 1
    assert drifts[0]["artifact"] == "row-000/attempt-00/.formal.tmp/oaves.csv"

    failed_row = row.model_copy(
        update={
            "status": CoverageStatus.FAILED,
            "attempts": [
                attempt_ok.model_copy(update={"status": CoverageStatus.FAILED})
            ],
        }
    )
    with pytest.raises(ValueError, match="coverage resume artifact drift"):
        c3_final_audit._verify_report_and_artifact_integrity(
            TargetCoverageReport(
                identity={"synthetic": True},
                identity_sha256=_FAKE_HASH,
                decisions=[],
                rows=[failed_row],
            ),
            expected,
            cov,
        )

    transient.write_text("before\n", encoding="utf-8")
    readiness.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="coverage resume artifact drift"):
        c3_final_audit._verify_report_and_artifact_integrity(
            report, expected, cov
        )


def test_c3_final_audit_maps_explicit_mapper_timeout_signature() -> None:
    attempt = _failed_attempt(
        stage="formal_analysis",
        reason_code="mapper_timeout",
        message="Orojenesis mapper exceeded 7200s timeout",
        formal=True,
    )
    attempt.status = CoverageStatus.BUDGET_EXHAUSTED
    attempt.wall_seconds = 7201.0
    row = _make_row(
        semantic_id="llm.attention.local_global.v2",
        slot="throughput-high",
        ir_path=IRPath.MAKE_FX_ATEN,
        family="attention",
        status=CoverageStatus.BUDGET_EXHAUSTED,
        reason="mapper_timeout",
        attempt=attempt,
    )
    cat, sub = c3_final_audit.classify_m0_reporting(row)
    assert (cat, sub) == ("budget_exhausted", "mapper_timeout")
    assert c3_final_audit._failure_signature(row, sub) == (
        "formal_analysis:mapper_timeout_7200s",
        "formal_analysis",
        "orojenesis_mapspace_pruning_and_timeout_code",
    )
