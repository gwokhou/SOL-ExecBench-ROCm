"""Unit tests for M3/D0 synthesis verification and bounded bundle emission."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.internal.solar import d0_synthesis
from scripts.internal.solar.d0_synthesis import (
    MAX_ARTIFACT_EXPECTATIONS,
    MAX_FOLLOWUP_ITEMS,
    MAX_REPORT_EXPECTATIONS,
    ArtifactExpectation,
    ClaimRecord,
    ClaimVerdict,
    D0Inputs,
    DeferredItem,
    FollowupTask,
    LineDomain,
    default_d0_inputs,
    main,
    synthesize_d0_bundle,
    verify_artifact_expectation,
)
from sol_execbench.core.integrity import sha256_file


def _sample_followup(task_id: str, order: int) -> FollowupTask:
    return FollowupTask(
        task_id=task_id,
        execution_order=order,
        priority="P1",
        title=f"Sample {task_id}",
        consumer="unit-test",
        owning_paths=("src/solar/",),
        problem_statement="Synthetic gap.",
        acceptance_criteria=("Pass unit checks.",),
        exposure_constraint="Keep unexposed holdouts untouched.",
    )


def _sample_inputs(
    *,
    receipts: tuple[ArtifactExpectation, ...],
    reports: tuple[ArtifactExpectation, ...],
    followups: tuple[FollowupTask, ...] | None = None,
    l3d_candidates: int = 0,
) -> D0Inputs:
    selected = (
        (
            _sample_followup("F3", 3),
            _sample_followup("F2", 1),
            _sample_followup("F1", 2),
        )
        if followups is None
        else followups
    )
    return D0Inputs(
        sealed_receipts=receipts,
        tracked_reports=reports,
        claims=(
            ClaimRecord(
                claim_id="B-1",
                domain=LineDomain.BASELINE,
                verdict=ClaimVerdict.PASSED,
                statement="Representative checks passed.",
                scope_boundary="Unit fixture only.",
                evidence_paths=("b2/gate.json",),
            ),
        ),
        followups=selected,
        deferred_items=(
            DeferredItem(
                item_id="DEF-L3D",
                title="L3-d",
                disposition="DEFERRED",
                reason="Zero common frontend candidates.",
            ),
        ),
        l3d_frontend_candidates_count=l3d_candidates,
    )


def test_default_d0_inputs_satisfy_m3_constraints() -> None:
    inputs = default_d0_inputs()
    d0_synthesis.validate_d0_inputs(inputs)
    assert len(inputs.sealed_receipts) == 47
    assert len(inputs.tracked_reports) == 14
    assert len(inputs.sealed_receipts) <= MAX_ARTIFACT_EXPECTATIONS
    assert len(inputs.tracked_reports) <= MAX_REPORT_EXPECTATIONS
    assert len(inputs.followups) == MAX_FOLLOWUP_ITEMS
    assert [task.task_id for task in inputs.followups] == ["F2", "F1", "F3"]
    assert inputs.l3d_frontend_candidates_count == 0
    assert {claim.domain for claim in inputs.claims} == {
        LineDomain.BASELINE,
        LineDomain.JUDGE,
        LineDomain.LLM_CORE,
    }


def test_synthesize_d0_bundle_writes_verified_bundle_deterministically(
    tmp_path: Path,
) -> None:
    receipt_rel = "b2/gate.json"
    receipt_path = tmp_path / "data" / "outputs" / receipt_rel
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text('{"ready": true}\n', encoding="utf-8")

    judge_rel = "e2/audit.json"
    judge_path = tmp_path / "data" / "outputs" / judge_rel
    judge_path.parent.mkdir(parents=True, exist_ok=True)
    judge_path.write_text('{"passed": true}\n', encoding="utf-8")

    report_rel = "docs/internal/sample-report.md"
    report_path = tmp_path / report_rel
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("# Sample\n", encoding="utf-8")

    inputs = _sample_inputs(
        receipts=(
            ArtifactExpectation(
                domain=LineDomain.BASELINE,
                relative_path=receipt_rel,
                expected_sha256=sha256_file(receipt_path),
                role="Baseline receipt",
            ),
            ArtifactExpectation(
                domain=LineDomain.JUDGE,
                relative_path=judge_rel,
                expected_sha256=sha256_file(judge_path),
                role="Judge receipt",
            ),
        ),
        reports=(
            ArtifactExpectation(
                domain=LineDomain.LLM_CORE,
                relative_path=report_rel,
                expected_sha256=sha256_file(report_path),
                role="Report",
            ),
        ),
    )

    out_dir = tmp_path / "out"
    first = synthesize_d0_bundle(
        repo_root=tmp_path,
        output_dir=out_dir,
        run_id="test-d0",
        inputs=inputs,
    )
    second = synthesize_d0_bundle(
        repo_root=tmp_path,
        output_dir=out_dir,
        run_id="test-d0",
        inputs=inputs,
    )

    assert first.all_referenced_artifacts_verified is True
    assert first.selected_count == 3
    assert first.execution_order == ("F2", "F1", "F3")
    assert first.emitted_file_digests == second.emitted_file_digests
    assert set(first.emitted_file_digests) == {
        "baseline/index.json",
        "judge/index.json",
        "llm-core/index.json",
        "decision/claims-matrix.json",
        "decision/selected-followups.json",
        "decision/d0-manifest.json",
    }

    followups_payload = json.loads(
        (out_dir / "decision" / "selected-followups.json").read_text(
            encoding="utf-8"
        )
    )
    assert followups_payload["hard_cap"] == 3
    assert followups_payload["execution_order"] == ["F2", "F1", "F3"]
    assert followups_payload["l3d_frontend_candidates_count"] == 0


def test_verify_artifact_expectation_rejects_missing_hash_traversal_and_oversize(
    tmp_path: Path,
) -> None:
    target = tmp_path / "artifact.json"
    target.write_text('{"ok": true}\n', encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="Missing referenced artifact"):
        verify_artifact_expectation(
            tmp_path,
            ArtifactExpectation(
                domain=LineDomain.BASELINE,
                relative_path="missing.json",
            ),
        )

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_artifact_expectation(
            tmp_path,
            ArtifactExpectation(
                domain=LineDomain.BASELINE,
                relative_path="artifact.json",
                expected_sha256="0" * 64,
            ),
        )

    with pytest.raises(ValueError, match="without '\\.\\.'"):
        verify_artifact_expectation(
            tmp_path,
            ArtifactExpectation(
                domain=LineDomain.BASELINE,
                relative_path="../escape.json",
            ),
        )

    with pytest.raises(ValueError, match="exceeds cap"):
        verify_artifact_expectation(
            tmp_path,
            ArtifactExpectation(
                domain=LineDomain.BASELINE,
                relative_path="artifact.json",
            ),
            max_bytes=4,
        )


@pytest.mark.parametrize(
    ("case_kind", "match"),
    [
        ("excess_receipts", "sealed_receipts count"),
        ("excess_reports", "tracked_reports count"),
        ("excess_followups", "hard cap 3"),
        ("non_contiguous_orders", "execution_order"),
        ("nonzero_l3d", "L3-d cannot stay deferred"),
    ],
    ids=[
        "excess-receipts",
        "excess-reports",
        "excess-followups",
        "non-contiguous-orders",
        "nonzero-l3d",
    ],
)
def test_validate_d0_inputs_rejects_boundary_violations(
    case_kind: str,
    match: str,
) -> None:
    one_receipt = ArtifactExpectation(
        domain=LineDomain.BASELINE,
        relative_path="x.json",
    )
    one_report = ArtifactExpectation(
        domain=LineDomain.BASELINE,
        relative_path="docs/x.md",
    )
    receipts: tuple[ArtifactExpectation, ...] = (one_receipt,)
    reports: tuple[ArtifactExpectation, ...] = (one_report,)
    followups: tuple[FollowupTask, ...] | None = None
    l3d_candidates = 0

    if case_kind == "excess_receipts":
        receipts = (one_receipt,) * (MAX_ARTIFACT_EXPECTATIONS + 1)
    elif case_kind == "excess_reports":
        reports = (one_report,) * (MAX_REPORT_EXPECTATIONS + 1)
    elif case_kind == "excess_followups":
        followups = tuple(
            _sample_followup(f"F{idx}", idx)
            for idx in range(1, MAX_FOLLOWUP_ITEMS + 2)
        )
    elif case_kind == "non_contiguous_orders":
        followups = (
            _sample_followup("F2", 1),
            _sample_followup("F1", 3),
        )
    elif case_kind == "nonzero_l3d":
        l3d_candidates = 1

    inputs = _sample_inputs(
        receipts=receipts,
        reports=reports,
        followups=followups,
        l3d_candidates=l3d_candidates,
    )
    with pytest.raises(ValueError, match=match):
        d0_synthesis.validate_d0_inputs(inputs)


def test_synthesize_d0_bundle_rejects_blank_run_id(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="run_id must be non-empty"):
        synthesize_d0_bundle(
            repo_root=tmp_path,
            output_dir=tmp_path / "out",
            run_id="   ",
            inputs=_sample_inputs(receipts=(), reports=()),
        )


def test_main_cli_prints_emitted_digests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        d0_synthesis,
        "default_d0_inputs",
        lambda: _sample_inputs(receipts=(), reports=()),
    )
    out_dir = tmp_path / "bundle"
    exit_code = main(
        [
            "--repo-root",
            str(tmp_path),
            "--run-id",
            "cli-test",
            "--output-dir",
            str(out_dir),
        ]
    )
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "decision/d0-manifest.json:" in captured.out
    assert (out_dir / "decision" / "d0-manifest.json").is_file()
