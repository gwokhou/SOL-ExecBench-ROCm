"""C1 metadata and scheduling tests with fake workers, never GPU or mapper."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
import yaml

from sol_execbench.core.data.json_utils import atomic_write_json_value
from sol_execbench.core.dataset.target_view import (
    load_materialized_target_view,
    require_llm_core_scope,
)
from sol_execbench.core.solar_bridge import target_coverage
from sol_execbench.core.solar_bridge.coverage_evidence import (
    classify_failure,
    verify_stage_outcome,
)
from sol_execbench.core.solar_bridge.coverage_models import (
    CoverageOptions,
    CoverageStatus,
)
from sol_execbench.core.solar_bridge.coverage_plan import (
    coverage_rows,
    load_coverage_target,
)
from sol_execbench.core.solar_bridge.models import (
    SolarReadinessStatus,
    SolarStage,
    SolarStageAuditOutcome,
)
from sol_execbench.core.solar_bridge.observation import StageObservation
from solar.observation import collect_stage_costs, observed_stage


@pytest.fixture(autouse=True)
def no_devices_or_mapper(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("C1 CPU checks must not enter GPU or mapper")

    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    monkeypatch.setattr(torch.cuda, "get_device_properties", forbidden)
    monkeypatch.setattr(target_coverage, "OrojenesisRunner", forbidden)
    monkeypatch.setattr(target_coverage, "run_solar_stage_worker", forbidden)
    monkeypatch.setattr(target_coverage, "run_solar_worker", forbidden)


def test_existing_view_reader_is_read_only(synthetic_target_files):
    source, view = synthetic_target_files
    before = {p: p.read_bytes() for p in view.parent.rglob("*") if p.is_file()}
    target = load_materialized_target_view(source, view)
    assert len(target.workloads) == 9
    rows = coverage_rows(target, "cuda:0")
    assert len(rows) == 18
    assert len({row.key for row in rows}) == 18
    assert [r.slot for r in rows[:2]] == ["smoke", "smoke"]
    assert [str(r.ir_path) for r in rows[:2]] == [
        "make_fx_aten",
        "torchview_extended_einsum",
    ]
    assert before == {
        p: p.read_bytes() for p in view.parent.rglob("*") if p.is_file()
    }
    with pytest.raises(ValueError, match="36 decisions"):
        require_llm_core_scope(target)
    with pytest.raises(ValueError, match="frozen"):
        load_coverage_target(source, view, source)


@pytest.mark.parametrize(
    "mutation",
    [
        "old-schema",
        "manifest-drift",
        "duplicate-uuid",
        "missing-slot",
        "partial",
        "path-escape",
        "hardware-conflict",
        "cohort-conflict",
        "rule-drift",
        "workload-drift",
        "decision-omission",
    ],
    ids=str,
)
def test_rejects_invalid_materialized_boundary(
    synthetic_target_files, mutation
):
    source, view = synthetic_target_files
    raw = yaml.safe_load(view.read_text())
    if mutation == "old-schema":
        raw["schema_version"] = "retired-test-version"
    elif mutation == "manifest-drift":
        raw["source_manifest_sha256"] = "0" * 64
    elif mutation == "duplicate-uuid":
        raw["workloads"][1]["uuid"] = raw["workloads"][0]["uuid"]
    elif mutation == "missing-slot":
        raw["workloads"].pop()
    elif mutation == "partial":
        raw["require_complete_profile"] = False
    elif mutation == "path-escape":
        raw["problems"][0]["definition_path"] = (
            "../source/problem/definition.json"
        )
    elif mutation == "hardware-conflict":
        raw["hardware_configuration_id"] = "1" * 64
    elif mutation == "cohort-conflict":
        raw["generation_cohort_id"] = "2" * 64
    elif mutation == "rule-drift":
        (source.parent / "problem/rule.yaml").write_text("changed")
    elif mutation == "workload-drift":
        (view.parent / "problem/workload.jsonl").write_text("{}\n")
    elif mutation == "decision-omission":
        raw["decisions"] = []
    view.write_text(yaml.safe_dump(raw))
    with pytest.raises((ValueError, KeyError)):
        load_materialized_target_view(source, view)


def test_unknown_failure_never_becomes_unsupported():
    assert (
        classify_failure("unfamiliar_reason", "ir_conversion")
        is CoverageStatus.FAILED
    )
    assert (
        classify_failure("worker_timeout", "outer_bridge")
        is CoverageStatus.BUDGET_EXHAUSTED
    )
    assert (
        classify_failure("unsupported_operation", "ir_conversion")
        is CoverageStatus.UNSUPPORTED
    )
    assert (
        classify_failure("worker_no_response", "outer_bridge")
        is CoverageStatus.INFRASTRUCTURE_ERROR
    )


def _fake_mapper(monkeypatch):
    monkeypatch.setattr(
        target_coverage,
        "OrojenesisRunner",
        lambda _: SimpleNamespace(
            toolchain_identity={"fixture": "synthetic, never executed"}
        ),
    )


def _unsupported(request, **kwargs):
    atomic_write_json_value(
        kwargs["observation_path"],
        StageObservation(
            records=[{"stage": "graph_extraction", "wall_seconds": 0.01}]
        ).model_dump(mode="json"),
    )
    return SolarStageAuditOutcome(
        status=SolarReadinessStatus.FAILED,
        analysis_id=request.workload_uuid,
        ir_path=request.ir_path,
        failure_stage=SolarStage.GRAPH_EXTRACTION,
        reason_code="unsupported_operation",
        message="synthetic unsupported fixture",
    )


def test_unsupported_rows_do_not_enter_formal_and_resume_binds_sources(
    synthetic_target_files, tmp_path, monkeypatch
):
    source, view = synthetic_target_files
    target = load_materialized_target_view(source, view)
    options = CoverageOptions(
        orojenesis_home=tmp_path / "synthetic-unused-mapper"
    )
    _fake_mapper(monkeypatch)
    calls = []

    def stage(request, **kwargs):
        calls.append((request.workload_uuid, request.ir_path))
        return _unsupported(request, **kwargs)

    monkeypatch.setattr(target_coverage, "run_solar_stage_worker", stage)
    result = target_coverage._audit_validated_target(
        target, tmp_path / "audit", options
    )
    assert len(calls) == 18
    assert all(len(r.attempts) == 1 for r in result.rows)
    assert all(
        r.status is CoverageStatus.UNSUPPORTED and r.attempts[0].formal is None
        for r in result.rows
    )
    resumed = target_coverage._audit_validated_target(
        target, tmp_path / "audit", options, resume=True
    )
    assert len(calls) == 18
    assert len(resumed.rows) == 18
    changed = options.model_copy(update={"readiness_seconds": 10})
    with pytest.raises(ValueError, match="resume identity"):
        target_coverage._audit_validated_target(
            target, tmp_path / "audit", changed, resume=True
        )
    artifact = next((tmp_path / "audit").rglob("readiness-cost.json"))
    artifact.write_text("tampered")
    with pytest.raises(ValueError, match="artifact drift"):
        target_coverage._audit_validated_target(
            target, tmp_path / "audit", options, resume=True
        )


def test_smoke_infrastructure_failure_keeps_all_rows(
    synthetic_target_files, tmp_path, monkeypatch
):
    source, view = synthetic_target_files
    target = load_materialized_target_view(source, view)
    _fake_mapper(monkeypatch)
    monkeypatch.setattr(
        target_coverage,
        "run_solar_stage_worker",
        lambda request, **kwargs: SolarStageAuditOutcome(
            status=SolarReadinessStatus.FAILED,
            analysis_id=request.workload_uuid,
            ir_path=request.ir_path,
            failure_stage=SolarStage.OUTER_BRIDGE,
            reason_code="worker_no_response",
        ),
    )
    result = target_coverage._audit_validated_target(
        target, tmp_path / "audit", CoverageOptions(orojenesis_home=tmp_path)
    )
    assert len(result.rows) == 18
    assert sum(bool(r.attempts) for r in result.rows) == 1
    assert not result.complete
    assert all(r.status is CoverageStatus.NOT_REACHED for r in result.rows[1:])


def test_cross_path_and_tampered_worker_artifacts_rejected(
    synthetic_target_files, tmp_path
):
    source, view = synthetic_target_files
    target = load_materialized_target_view(source, view)
    row, other = coverage_rows(target, "cpu")[:2]
    outcome = SolarStageAuditOutcome(
        status=SolarReadinessStatus.FAILED,
        analysis_id=row.workload_uuid,
        ir_path=other.ir_path,
        failure_stage=SolarStage.IR_CONVERSION,
    )
    with pytest.raises(ValueError, match="identity"):
        verify_stage_outcome(target, row, outcome, tmp_path, "cpu")
    (tmp_path / "artifact").write_text("actual")
    outcome = replace(
        outcome,
        ir_path=row.ir_path,
        output_dir=str(tmp_path),
        artifacts=({"path": "artifact", "sha256": "a" * 64},),
    )
    with pytest.raises(ValueError, match="checksum"):
        verify_stage_outcome(target, row, outcome, tmp_path, "cpu")


def test_observation_is_opt_in_and_records_exception(monkeypatch):
    @observed_stage(SolarStage.FORMAL_ANALYSIS)
    def function():
        raise ValueError("synthetic failure")

    records = []
    with pytest.raises(ValueError, match="synthetic"):
        function()
    assert records == []
    with (
        collect_stage_costs(records),
        pytest.raises(ValueError, match="synthetic"),
    ):
        function()
    assert len(records) == 1
    assert records[0]["wall_seconds"] >= 0
    assert records[0]["host_rss_sampled_peak_bytes"] > 0
    assert records[0]["device_peak_bytes"] is None


def test_full_synthetic_matrix_has_frozen_role_denominators(
    synthetic_target_files,
):
    from collections import Counter

    source, view = synthetic_target_files
    target = load_materialized_target_view(source, view)
    records, entries = [], []
    definitions, workloads = {}, {}
    for index in range(20):
        semantic_id = f"synthetic.scheduler.{index:02d}"
        entries.append(
            target.source.entries[0].model_copy(
                update={"semantic_id": semantic_id}
            )
        )
        definitions[semantic_id] = next(iter(target.definitions.values()))
        for record in target.view.workloads:
            uuid = f"synthetic-{index}-{record.uuid}"
            records.append(
                record.model_copy(
                    update={"semantic_id": semantic_id, "uuid": uuid}
                )
            )
            workloads[uuid] = target.workloads[record.uuid].model_copy(
                update={"uuid": uuid}
            )
    matrix_fixture = replace(
        target,
        source=target.source.model_copy(update={"entries": tuple(entries)}),
        view=target.view.model_copy(update={"workloads": tuple(records)}),
        definitions=definitions,
        workloads=workloads,
    )
    rows = coverage_rows(matrix_fixture, "cpu")
    assert len(rows) == len({row.key for row in rows}) == 360
    assert Counter(row.role for row in rows) == {
        "smoke": 40,
        "development": 160,
        "holdout": 160,
    }
    assert all(row.slot == "smoke" for row in rows[:40])


def test_only_ready_enters_formal(
    synthetic_target_files, tmp_path, monkeypatch
):
    from sol_execbench.core.solar_bridge.models import (
        SolarAnalysisOutcome,
        SolarAnalysisStatus,
        readiness_stage_artifacts,
    )

    source, view = synthetic_target_files
    target = load_materialized_target_view(source, view)
    row = coverage_rows(target, "cpu")[0]
    _fake_mapper(monkeypatch)
    calls = []

    def ready(request, **kwargs):
        stages = tuple(
            {
                "stage": stage,
                "status": "passed",
                "artifact": {"path": path, "sha256": "a" * 64},
            }
            for stage, path in readiness_stage_artifacts(
                request.ir_path
            ).items()
        )
        return SolarStageAuditOutcome(
            status=SolarReadinessStatus.READY,
            analysis_id=row.workload_uuid,
            output_dir=request.output_dir,
            ir_path=request.ir_path,
            architecture_sha256="a" * 64,
            stages=stages,
        )

    def formal(request, **kwargs):
        calls.append(request)
        return SolarAnalysisOutcome(
            status=SolarAnalysisStatus.FAILED,
            analysis_id=row.workload_uuid,
            ir_path=request.ir_path,
            stage=SolarStage.FORMAL_ANALYSIS,
            reason_code="formal_bound_unproved",
        )

    monkeypatch.setattr(target_coverage, "run_solar_stage_worker", ready)
    monkeypatch.setattr(target_coverage, "run_solar_worker", formal)
    # This test isolates dispatch; other tests exercise actual receipt/hash rejection.
    monkeypatch.setattr(
        target_coverage, "verify_stage_outcome", lambda *args: {}
    )
    monkeypatch.setattr(
        target_coverage,
        "verification_dimensions",
        lambda *args: {"fixture": True},
    )
    directory = tmp_path / "audit/row-000/attempt-00"
    directory.mkdir(parents=True)
    attempt = target_coverage._audit_row(
        target, row, directory, CoverageOptions(orojenesis_home=tmp_path), 10
    )
    assert len(calls) == 1
    assert calls[0].ir_path is row.ir_path
    assert attempt.status is CoverageStatus.UNPROVED
