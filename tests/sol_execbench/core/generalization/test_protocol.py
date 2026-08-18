from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import yaml
from click.testing import CliRunner
from pydantic import BaseModel
from sol_execbench_type_helpers import make_solution, make_trace, make_workload

from sol_execbench.cli.main import cli
from sol_execbench.core.data.json_utils import atomic_write_json_value
from sol_execbench.core.dataset.corpus import (
    generate_corpus,
    load_corpus_manifest,
    load_target_descriptor,
)
from sol_execbench.core.dataset.corpus_models import (
    CorpusManifest,
    CorpusProfile,
    CorpusTargetViewManifest,
    WorkloadRole,
)
from sol_execbench.core.generalization.metrics import workload_drift
from sol_execbench.core.generalization.models import (
    CandidateDeclaration,
    CorpusAgentView,
    GeneralizationReportStatus,
    HardwareContextView,
    HardwareGeneralizationCell,
    HardwareGeneralizationPlan,
    HardwareGeneralizationReport,
    HardwareShift,
    TrainingExposureDeclaration,
    TrainingHardwareExposure,
)
from sol_execbench.core.generalization.workflow import (
    PlannedStudy,
    aggregate_study,
    build_study_plan,
    seal_cell,
)
from sol_execbench.core.integrity import sha256_file, stable_json_checksum
from sol_execbench.core.platform.memory_quota import (
    GPUMemoryQuotaEvidence,
    capacity_probe_digest,
)
from sol_execbench.core.platform.schema_versions import PlatformArtifactSchema

ROOT = Path(__file__).resolve().parents[4]
RELEASE = ROOT / "problems/LLM_CORE/releases/LLM_CORE_V2"
MANIFEST = RELEASE / "manifest.yaml"
TARGETS = ROOT / "problems/LLM_CORE/targets"
GIB = 1024**3


@pytest.fixture(scope="module")
def corpus() -> CorpusManifest:
    return load_corpus_manifest(MANIFEST)


@pytest.fixture(scope="module")
def target_views(tmp_path_factory) -> dict[str, CorpusTargetViewManifest]:
    root = tmp_path_factory.mktemp("generalization-views")
    scenarios = (
        ("gfx1200-8", "gfx1200", 8),
        ("gfx1200-16", "gfx1200", 16),
        ("gfx942", "gfx942", 192),
    )
    views = {}
    for target_id, gfx_target, usable_gib in scenarios:
        output = root / target_id
        generate_corpus(
            MANIFEST,
            output,
            target=load_target_descriptor(TARGETS / f"{gfx_target}.yaml"),
            profiles=(CorpusProfile.CORE,),
            capacity_evidence=_capacity(usable_gib, gfx_target),
        )
        views[target_id] = _view(output)
    return views


@pytest.fixture(scope="module")
def planned(
    corpus: CorpusManifest,
    target_views: dict[str, CorpusTargetViewManifest],
) -> PlannedStudy:
    exposure = TrainingExposureDeclaration(
        hardware=(
            TrainingHardwareExposure(
                gfx_target="gfx1200",
                capacity_class_bytes=8 * GIB,
                distribution_id=target_views["gfx1200-8"].distribution_id,
            ),
        ),
    )
    return build_study_plan(
        study_id="mock-three-targets",
        manifest=corpus,
        manifest_digest=sha256_file(MANIFEST),
        exposure=exposure,
        targets=tuple(target_views.items()),
    )


def _capacity(usable_gib: int, gfx_target: str) -> GPUMemoryQuotaEvidence:
    raw = usable_gib * GIB * 5 // 4
    payload: dict[str, Any] = {
        "schema_version": PlatformArtifactSchema.GPU_MEMORY_QUOTA_EVIDENCE,
        "device": "cuda:0",
        "device_index": 0,
        "gpu_name": f"mock {gfx_target}",
        "gfx_target": gfx_target,
        "torch_version": "test",
        "hip_version": "test",
        "collected_at": datetime(2026, 8, 17, tzinfo=UTC),
        "runtime_free_bytes": raw,
        "runtime_total_bytes": raw,
        "environment_quota_bytes": None,
        "stable_allocatable_bytes": raw,
        "harness_reserve_bytes": 0,
        "safety_percent": 80,
        "usable_budget_bytes": usable_gib * GIB,
        "capacity_probe_digest": "0" * 64,
    }
    provisional = GPUMemoryQuotaEvidence.model_construct(**payload)
    payload["capacity_probe_digest"] = capacity_probe_digest(provisional)
    return GPUMemoryQuotaEvidence.model_validate(payload)


def _view(root: Path) -> CorpusTargetViewManifest:
    raw = yaml.safe_load(
        (root / "target-view-manifest.yaml").read_text(encoding="utf-8")
    )
    return CorpusTargetViewManifest.model_validate(raw)


def _cells(
    planned: PlannedStudy,
    corpus: CorpusManifest,
    target_views: dict[str, CorpusTargetViewManifest],
):
    return tuple(
        seal_cell(
            plan=planned.plan,
            cell_id=cell.cell_id,
            target_view=target_views[cell.study_target_id],
            manifest=corpus,
            manifest_digest=sha256_file(MANIFEST),
            solutions=(),
            traces=(),
            observed_gfx_target=(
                target_views[cell.study_target_id].target.gfx_target
            ),
            observed_capacity_class_bytes=(
                target_views[cell.study_target_id].capacity_class_bytes
            ),
        )
        for cell in planned.plan.cells
    )


def _solution_and_trace(
    corpus: CorpusManifest,
    target_view: CorpusTargetViewManifest,
    *,
    trace_solution: str = "candidate",
    axes: dict[str, int] | None = None,
    workload_uuid: str | None = None,
):
    workload = next(
        item
        for item in target_view.workloads
        if item.role is not WorkloadRole.SMOKE
    )
    entry = next(
        item
        for item in corpus.entries
        if item.semantic_id == workload.semantic_id
    )
    solution = make_solution(
        name="candidate",
        definition=entry.problem_name,
        author="test-agent",
        spec={
            "languages": ["pytorch"],
            "target_hardware": ["LOCAL"],
            "entry_point": "kernel.py::run",
        },
        sources=[{"path": "kernel.py", "content": "def run(*args): pass\n"}],
    )
    trace = make_trace(
        definition=entry.problem_name,
        solution=trace_solution,
        workload=make_workload(
            uuid=workload_uuid or workload.uuid,
            axes=axes if axes is not None else workload.axes,
            inputs={"x": {"type": "random"}},
        ),
        evaluation={
            "status": "COMPILE_ERROR",
            "environment": {
                "hardware": target_view.target.gfx_target,
                "libs": {},
            },
            "timestamp": "2026-08-17T00:00:00Z",
        },
    )
    return solution, trace


def test_roles_and_agent_visibility_are_exact(
    planned: PlannedStudy,
    target_views: dict[str, CorpusTargetViewManifest],
) -> None:
    for target_id, target_view in target_views.items():
        grouped: dict[str, list[Any]] = {}
        for workload in target_view.workloads:
            grouped.setdefault(workload.semantic_id, []).append(workload)
        for records in grouped.values():
            roles = [item.role for item in records]
            assert roles.count(WorkloadRole.SMOKE) == 1
            assert roles.count(WorkloadRole.DEVELOPMENT) == 4
            assert roles.count(WorkloadRole.HOLDOUT) == 4

        agent_view = planned.agent_views[f"{target_id}--full_facts"]
        for definition in agent_view.definitions:
            assert len(definition.development_workloads) == 4
            assert len(definition.withheld_slot_ids) == 4
            assert all(
                item.role is WorkloadRole.DEVELOPMENT
                for item in definition.development_workloads
            )
            serialized = definition.model_dump_json()
            holdout_uuids = {
                item.uuid
                for item in target_view.workloads
                if item.semantic_id == definition.semantic_id
                and item.role is WorkloadRole.HOLDOUT
            }
            assert not any(uuid in serialized for uuid in holdout_uuids)


@pytest.mark.parametrize(
    "model",
    (
        CorpusAgentView,
        HardwareGeneralizationPlan,
        HardwareGeneralizationCell,
        HardwareGeneralizationReport,
    ),
)
def test_artifact_readers_require_exact_current_schema(
    model: type[BaseModel],
) -> None:
    with pytest.raises(ValueError, match="requires schema_version"):
        model.model_validate({})


def test_mock_targets_are_classified_from_exposure(
    planned: PlannedStudy,
) -> None:
    shifts = {cell.study_target_id: cell.shift for cell in planned.plan.cells}
    assert shifts["gfx1200-8"] is HardwareShift.SEEN_HARDWARE_SEEN_CAPACITY
    assert shifts["gfx1200-16"] is HardwareShift.SEEN_ARCHITECTURE_NEW_CAPACITY
    assert shifts["gfx942"] is HardwareShift.UNSEEN_ARCHITECTURE


def test_core_matrix_is_small_and_anonymous_is_optional(
    planned: PlannedStudy,
    corpus: CorpusManifest,
    target_views: dict[str, CorpusTargetViewManifest],
) -> None:
    assert len(planned.plan.cells) == 6
    assert {cell.context_view for cell in planned.plan.cells} == {
        HardwareContextView.FULL_FACTS
    }

    expanded = build_study_plan(
        study_id="anonymous-ablation",
        manifest=corpus,
        manifest_digest=sha256_file(MANIFEST),
        exposure=planned.plan.exposure,
        targets=tuple(target_views.items()),
        include_anonymous=True,
    )
    anonymous = [
        view
        for view in expanded.agent_views.values()
        if view.hardware_facts.context_view
        is HardwareContextView.ANONYMIZED_FACTS
    ]
    assert len(expanded.plan.cells) == 9
    assert anonymous
    assert all(view.hardware_facts.gfx_target is None for view in anonymous)
    assert all(view.hardware_facts.target_id is None for view in anonymous)
    target_names = {
        target.target.gfx_target for target in target_views.values()
    } | {target.target.target_id for target in target_views.values()}
    for view in anonymous:
        serialized = view.model_dump_json()
        assert not any(name in serialized for name in target_names)
        assert view.hardware_facts.study_target_id.startswith("target-")


def test_slot_signatures_align_while_realizations_may_change(
    corpus: CorpusManifest,
    target_views: dict[str, CorpusTargetViewManifest],
) -> None:
    first = target_views["gfx1200-8"]
    for target_id in ("gfx1200-16", "gfx942"):
        drift = workload_drift(
            corpus,
            first,
            target_views[target_id],
            source_target_id="gfx1200-8",
            target_target_id=target_id,
        )
        assert drift.latent_slot_signature_equal
        assert drift.common_definition_count <= drift.target_definition_count
        assert 0.0 <= drift.support_jaccard <= 1.0


def test_aggregate_is_deterministic_and_uses_common_support(
    planned: PlannedStudy,
    corpus: CorpusManifest,
    target_views: dict[str, CorpusTargetViewManifest],
) -> None:
    cells = _cells(planned, corpus, target_views)
    first = aggregate_study(
        plan=planned.plan,
        manifest=corpus,
        manifest_digest=sha256_file(MANIFEST),
        target_views=target_views,
        cells=cells,
    )
    second = aggregate_study(
        plan=planned.plan,
        manifest=corpus,
        manifest_digest=sha256_file(MANIFEST),
        target_views=target_views,
        cells=cells,
    )

    assert first == second
    assert first.status is GeneralizationReportStatus.COMPLETE
    assert first.generalization_conclusion_allowed
    assert first.report_digest == second.report_digest
    assert all(
        metric.correctness_rate.value == 0.0
        for metric in first.target_full.values()
    )
    common_counts = {
        metric.definition_count for metric in first.common_support.values()
    }
    assert len(common_counts) == 1
    assert "native_retention" not in first.model_dump(mode="json")


def test_capability_missingness_changes_only_target_full_denominator(
    tmp_path: Path,
    corpus: CorpusManifest,
    target_views: dict[str, CorpusTargetViewManifest],
) -> None:
    limited_target = load_target_descriptor(TARGETS / "gfx942.yaml").model_copy(
        update={"capabilities": ()}
    )
    output = tmp_path / "limited-gfx942"
    generate_corpus(
        MANIFEST,
        output,
        target=limited_target,
        profiles=(CorpusProfile.CORE,),
        capacity_evidence=_capacity(192, "gfx942"),
    )
    limited = _view(output)
    control = target_views["gfx1200-8"]
    exposure = TrainingExposureDeclaration(
        hardware=(
            TrainingHardwareExposure(
                gfx_target="gfx1200",
                capacity_class_bytes=8 * GIB,
                distribution_id=control.distribution_id,
            ),
        )
    )
    views = {"control": control, "limited": limited}
    study = build_study_plan(
        study_id="capability-missingness",
        manifest=corpus,
        manifest_digest=sha256_file(MANIFEST),
        exposure=exposure,
        targets=tuple(views.items()),
    )
    report = aggregate_study(
        plan=study.plan,
        manifest=corpus,
        manifest_digest=sha256_file(MANIFEST),
        target_views=views,
        cells=_cells(study, corpus, views),
    )

    full_counts = {
        cell_id: metrics.definition_count
        for cell_id, metrics in report.target_full.items()
    }
    assert (
        full_counts["limited--target_conditioned--full_facts"]
        < full_counts["control--target_conditioned--full_facts"]
    )
    assert (
        len(
            {
                metrics.definition_count
                for metrics in report.common_support.values()
            }
        )
        == 1
    )
    assert report.workload_drift[0].skip_reason_counts["missing_capability"] > 0


def test_missing_cell_produces_no_generalization_conclusion(
    planned: PlannedStudy,
    corpus: CorpusManifest,
    target_views: dict[str, CorpusTargetViewManifest],
) -> None:
    cells = _cells(planned, corpus, target_views)[:-1]
    report = aggregate_study(
        plan=planned.plan,
        manifest=corpus,
        manifest_digest=sha256_file(MANIFEST),
        target_views=target_views,
        cells=cells,
    )

    assert report.status is GeneralizationReportStatus.INCOMPLETE
    assert not report.generalization_conclusion_allowed
    assert len(report.missing_cell_ids) == 1


def test_portability_payload_change_is_rejected(
    planned: PlannedStudy,
    corpus: CorpusManifest,
    target_views: dict[str, CorpusTargetViewManifest],
) -> None:
    cells = list(_cells(planned, corpus, target_views))
    portable = [
        index
        for index, cell in enumerate(planned.plan.cells)
        if cell.track.value == "solution_portability"
    ]
    first_two_views = [
        target_views[planned.plan.cells[index].study_target_id]
        for index in portable[:2]
    ]
    semantic_id = min(
        {item.semantic_id for item in first_two_views[0].workloads}
        & {item.semantic_id for item in first_two_views[1].workloads}
    )
    for offset, index in enumerate(portable[:2]):
        planned_cell = planned.plan.cells[index]
        candidate = CandidateDeclaration(
            semantic_id=semantic_id,
            solution_digest="b" * 64,
            portability_digest=("c" if offset == 0 else "d") * 64,
            agent_view_digest=planned_cell.agent_view_digest,
            hardware_context_digest=planned_cell.hardware_context_digest,
        )
        payload = cells[index].model_dump(mode="json")
        payload["candidates"] = [candidate.model_dump(mode="json")]
        missing = []
        for result in payload["results"]:
            if result["semantic_id"] == semantic_id:
                result["status"] = "evaluator_failure"
                missing.append(f"missing_trace:{result['workload_uuid']}")
        payload["evaluator_failures"] = missing
        payload.pop("cell_digest")
        payload["cell_digest"] = stable_json_checksum(payload)
        cells[index] = type(cells[index]).model_validate(payload)

    with pytest.raises(ValueError, match="portability payload changed"):
        aggregate_study(
            plan=planned.plan,
            manifest=corpus,
            manifest_digest=sha256_file(MANIFEST),
            target_views=target_views,
            cells=tuple(cells),
        )


def test_runtime_sealing_never_calls_workload_generator(
    planned: PlannedStudy,
    corpus: CorpusManifest,
    target_views: dict[str, CorpusTargetViewManifest],
    monkeypatch,
) -> None:
    def fail_generation(*_args, **_kwargs):
        raise AssertionError("runtime failure attempted workload regeneration")

    monkeypatch.setattr(
        "sol_execbench.core.dataset.corpus.generate_rule_workloads",
        fail_generation,
    )
    cell = planned.plan.cells[0]
    sealed = seal_cell(
        plan=planned.plan,
        cell_id=cell.cell_id,
        target_view=target_views[cell.study_target_id],
        manifest=corpus,
        manifest_digest=sha256_file(MANIFEST),
        solutions=(),
        traces=(),
        observed_gfx_target="gfx1200",
        observed_capacity_class_bytes=(
            target_views[cell.study_target_id].capacity_class_bytes
        ),
    )
    expected = sum(
        item.role is not WorkloadRole.SMOKE
        for item in target_views[cell.study_target_id].workloads
    )
    assert len(sealed.results) == expected


def test_seal_rejects_manifest_and_runtime_capacity_drift(
    planned: PlannedStudy,
    corpus: CorpusManifest,
    target_views: dict[str, CorpusTargetViewManifest],
) -> None:
    cell = planned.plan.cells[0]
    target = target_views[cell.study_target_id]

    with pytest.raises(ValueError, match="corpus manifest differs"):
        seal_cell(
            plan=planned.plan,
            cell_id=cell.cell_id,
            target_view=target,
            manifest=corpus,
            manifest_digest="f" * 64,
            solutions=(),
            traces=(),
            observed_gfx_target=target.target.gfx_target,
            observed_capacity_class_bytes=target.capacity_class_bytes,
        )
    with pytest.raises(ValueError, match="observed capacity class differs"):
        seal_cell(
            plan=planned.plan,
            cell_id=cell.cell_id,
            target_view=target,
            manifest=corpus,
            manifest_digest=sha256_file(MANIFEST),
            solutions=(),
            traces=(),
            observed_gfx_target=target.target.gfx_target,
            observed_capacity_class_bytes=target.capacity_class_bytes * 2,
        )


def test_run_cell_cli_reprobes_runtime_capacity_class(
    tmp_path: Path,
    monkeypatch,
    planned: PlannedStudy,
    target_views: dict[str, CorpusTargetViewManifest],
) -> None:
    cell = next(
        item
        for item in planned.plan.cells
        if item.study_target_id == "gfx1200-8"
    )
    plan_path = tmp_path / "plan.json"
    target_path = tmp_path / "target.json"
    output = tmp_path / "cell.json"
    atomic_write_json_value(plan_path, planned.plan.model_dump(mode="json"))
    atomic_write_json_value(
        target_path,
        target_views["gfx1200-8"].model_dump(mode="json"),
    )
    monkeypatch.setattr(
        "sol_execbench.cli.commands.generalization."
        "collect_gpu_memory_quota_isolated",
        lambda *_args, **_kwargs: _capacity(16, "gfx1200"),
    )

    result = CliRunner().invoke(
        cli,
        [
            "--format",
            "json",
            "generalization",
            "run-cell",
            "--plan",
            str(plan_path),
            "--manifest",
            str(MANIFEST),
            "--target-view",
            str(target_path),
            "--cell-id",
            cell.cell_id,
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 2
    response = json.loads(result.output)
    assert response["error"]["code"] == "generalization_protocol_invalid"
    assert "observed capacity class differs" in response["error"]["message"]
    assert not output.exists()


@pytest.mark.parametrize(
    ("trace_overrides", "message"),
    (
        ({"trace_solution": "other"}, "trace Solution differs"),
        ({"axes": {"tampered": 1}}, "trace workload axes differ"),
        ({"workload_uuid": "unplanned"}, "unplanned workload traces"),
    ),
)
def test_seal_rejects_trace_identity_drift(
    trace_overrides: dict[str, Any],
    message: str,
    planned: PlannedStudy,
    corpus: CorpusManifest,
    target_views: dict[str, CorpusTargetViewManifest],
) -> None:
    cell = planned.plan.cells[0]
    target = target_views[cell.study_target_id]
    solution, trace = _solution_and_trace(
        corpus,
        target,
        **trace_overrides,
    )

    with pytest.raises(ValueError, match=message):
        seal_cell(
            plan=planned.plan,
            cell_id=cell.cell_id,
            target_view=target,
            manifest=corpus,
            manifest_digest=sha256_file(MANIFEST),
            solutions=(solution,),
            traces=(trace,),
            observed_gfx_target=target.target.gfx_target,
            observed_capacity_class_bytes=target.capacity_class_bytes,
        )


def test_aggregate_revalidates_cell_results_against_target_view(
    planned: PlannedStudy,
    corpus: CorpusManifest,
    target_views: dict[str, CorpusTargetViewManifest],
) -> None:
    cells = list(_cells(planned, corpus, target_views))
    payload = cells[0].model_dump(mode="json")
    payload["results"][0]["slot_id"] = "tampered-slot"
    payload.pop("cell_digest")
    payload["cell_digest"] = stable_json_checksum(payload)
    cells[0] = HardwareGeneralizationCell.model_validate(payload)

    with pytest.raises(ValueError, match="result metadata differs"):
        aggregate_study(
            plan=planned.plan,
            manifest=corpus,
            manifest_digest=sha256_file(MANIFEST),
            target_views=target_views,
            cells=tuple(cells),
        )


def test_aggregate_rejects_ambiguous_seen_control(
    corpus: CorpusManifest,
    target_views: dict[str, CorpusTargetViewManifest],
) -> None:
    control = target_views["gfx1200-8"]
    views = {
        "control-a": control,
        "control-b": control,
        "gfx942": target_views["gfx942"],
    }
    exposure = TrainingExposureDeclaration(
        hardware=(
            TrainingHardwareExposure(
                gfx_target="gfx1200",
                capacity_class_bytes=8 * GIB,
                distribution_id=control.distribution_id,
            ),
        )
    )
    study = build_study_plan(
        study_id="ambiguous-control",
        manifest=corpus,
        manifest_digest=sha256_file(MANIFEST),
        exposure=exposure,
        targets=tuple(views.items()),
    )

    with pytest.raises(ValueError, match="comparison control is ambiguous"):
        aggregate_study(
            plan=study.plan,
            manifest=corpus,
            manifest_digest=sha256_file(MANIFEST),
            target_views=views,
            cells=_cells(study, corpus, views),
        )


def test_plan_cli_emits_machine_readable_contract(
    tmp_path: Path,
    target_views: dict[str, CorpusTargetViewManifest],
) -> None:
    target_paths = []
    for target_id, view in target_views.items():
        path = tmp_path / f"{target_id}.json"
        atomic_write_json_value(path, view.model_dump(mode="json"))
        target_paths.extend(
            ("--target-id", target_id, "--target-view", str(path))
        )
    control = target_views["gfx1200-8"]
    seen = f"gfx1200:{8 * GIB}:{control.distribution_id}"
    output = tmp_path / "study"
    result = CliRunner().invoke(
        cli,
        [
            "--format",
            "json",
            "generalization",
            "plan",
            "--study-id",
            "cli-mock",
            "--seen-hardware",
            seen,
            "--manifest",
            str(MANIFEST),
            *target_paths,
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    response = json.loads(result.output)
    assert response["ok"] is True
    assert response["data"]["cells"] == 6
    assert (output / "plan.json").is_file()
    assert len(tuple(output.glob("*.agent-view.json"))) == 3
