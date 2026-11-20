"""CPU unit tests for the C2 target-view and C3 smoke campaign helpers."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml

from scripts.internal.solar import c2_c3_campaign, c2_c3_host
from sol_execbench.core.data.definition import Definition
from sol_execbench.core.data.json_utils import atomic_write_json_value
from sol_execbench.core.dataset.corpus import (
    generate_corpus,
    load_target_descriptor,
    semantic_fingerprint,
)
from sol_execbench.core.dataset.corpus_models import CorpusProfile
from sol_execbench.core.dataset.target_view import load_materialized_target_view
from sol_execbench.core.integrity import sha256_file
from sol_execbench.core.platform.memory_quota import (
    GPUMemoryQuotaEvidence,
    capacity_probe_digest,
    derive_usable_budget,
)
from sol_execbench.core.platform.schema_versions import PlatformArtifactSchema
from sol_execbench.core.solar_bridge import coverage_evidence, target_coverage
from sol_execbench.core.solar_bridge.coverage_models import (
    CoverageOptions,
    CoverageStatus,
)
from sol_execbench.core.solar_bridge.models import (
    SolarAnalysisOutcome,
    SolarAnalysisStatus,
    SolarReadinessStatus,
    SolarStage,
    SolarStageAuditOutcome,
    readiness_stage_artifacts,
)
from sol_execbench.core.solar_bridge.observation import StageObservation

ROOT = Path(__file__).resolve().parents[2]
RELEASE = ROOT / "problems/LLM_CORE/releases/LLM_CORE_V2"


def _build_synthetic_target_view(tmp_path: Path):
    source_root = tmp_path / "source"
    raw = yaml.safe_load((RELEASE / "manifest.yaml").read_text())
    entry = raw["entries"][0]
    definition = Definition.model_validate_json(
        (RELEASE / entry["definition_path"]).read_text()
    )
    rule = yaml.safe_load((RELEASE / entry["generation_rule_path"]).read_text())
    entry["semantic_id"] = rule["semantic_id"] = "synthetic.smoke.matmul"
    entry["definition_path"] = "problem/definition.json"
    entry["generation_rule_path"] = "problem/rule.yaml"
    source_id = entry["source_ids"][0]
    raw["sources"] = [
        item for item in raw["sources"] if item["source_id"] == source_id
    ]
    raw["sources"][0]["architecture_facts"]["hidden_size"] = 64
    raw["sources"][0]["architecture_facts"]["intermediate_size"] = 64
    for index, slot in enumerate(rule["slots"], 1):
        slot.update(scale_numerator=index, scale_denominator=1, irregular=False)
    (source_root / "problem").mkdir(parents=True)
    atomic_write_json_value(
        source_root / entry["definition_path"],
        definition.model_dump(mode="json"),
        sort_keys=False,
    )
    (source_root / entry["generation_rule_path"]).write_text(
        yaml.safe_dump(rule)
    )
    entry["definition_sha256"] = sha256_file(
        source_root / entry["definition_path"]
    )
    entry["generation_rule_sha256"] = sha256_file(
        source_root / entry["generation_rule_path"]
    )
    entry["semantic_fingerprint"] = semantic_fingerprint(definition, entry)
    raw["entries"] = [entry]
    raw["corpus_id"] = "synthetic_smoke_fixture"
    raw["coverage_policy"]["definition_count"] = 1
    raw["coverage_policy"]["operation_definition_counts"] = {
        key: int(key == "linear")
        for key in raw["coverage_policy"]["operation_definition_counts"]
    }
    raw["coverage_policy"]["profile_minimum_generated_definitions"] = {
        "core": 1
    }
    manifest_path = source_root / "manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(raw))
    payload = {
        "schema_version": PlatformArtifactSchema.GPU_MEMORY_QUOTA_EVIDENCE,
        "device": "cuda:0",
        "device_index": 0,
        "gpu_name": "synthetic CPU fixture, not hardware",
        "gfx_target": "gfx1200",
        "torch_version": "synthetic",
        "hip_version": "synthetic",
        "collected_at": datetime(2026, 1, 1, tzinfo=UTC),
        "runtime_free_bytes": 8 * 1024**3,
        "runtime_total_bytes": 8 * 1024**3,
        "environment_quota_bytes": None,
        "stable_allocatable_bytes": 8 * 1024**3,
        "harness_reserve_bytes": 0,
        "safety_percent": 50,
        "usable_budget_bytes": 4 * 1024**3,
        "capacity_probe_digest": "0" * 64,
    }
    payload["capacity_probe_digest"] = capacity_probe_digest(
        GPUMemoryQuotaEvidence.model_construct(
            _fields_set=set(payload), **payload
        )
    )
    capacity = GPUMemoryQuotaEvidence.model_validate(payload)
    target_desc = load_target_descriptor(
        ROOT / "problems/LLM_CORE/targets/isa/gfx1200.yaml"
    )
    view_dir = generate_corpus(
        manifest_path,
        tmp_path / "target-view",
        target=target_desc,
        profiles=(CorpusProfile.CORE,),
        capacity_evidence=capacity,
        require_complete_profile=True,
    )
    return load_materialized_target_view(
        manifest_path, view_dir / "target-view-manifest.yaml"
    )


def test_standard_16gb_roundtrips_real_rx9060xt_quota_observation(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    raw = yaml.safe_load((RELEASE / "manifest.yaml").read_text())
    entry = raw["entries"][0]
    definition = Definition.model_validate_json(
        (RELEASE / entry["definition_path"]).read_text()
    )
    rule = yaml.safe_load((RELEASE / entry["generation_rule_path"]).read_text())
    entry["semantic_id"] = rule["semantic_id"] = "synthetic.rx9060xt.matmul"
    entry["definition_path"] = "problem/definition.json"
    entry["generation_rule_path"] = "problem/rule.yaml"
    source_id = entry["source_ids"][0]
    raw["sources"] = [
        item for item in raw["sources"] if item["source_id"] == source_id
    ]
    raw["sources"][0]["architecture_facts"]["hidden_size"] = 64
    raw["sources"][0]["architecture_facts"]["intermediate_size"] = 64
    for index, slot in enumerate(rule["slots"], 1):
        slot.update(scale_numerator=index, scale_denominator=1, irregular=False)
    (source_root / "problem").mkdir(parents=True)
    atomic_write_json_value(
        source_root / entry["definition_path"],
        definition.model_dump(mode="json"),
        sort_keys=False,
    )
    (source_root / entry["generation_rule_path"]).write_text(
        yaml.safe_dump(rule)
    )
    entry["definition_sha256"] = sha256_file(
        source_root / entry["definition_path"]
    )
    entry["generation_rule_sha256"] = sha256_file(
        source_root / entry["generation_rule_path"]
    )
    entry["semantic_fingerprint"] = semantic_fingerprint(definition, entry)
    raw["entries"] = [entry]
    raw["corpus_id"] = "synthetic_rx9060xt_roundtrip"
    raw["coverage_policy"]["definition_count"] = 1
    raw["coverage_policy"]["operation_definition_counts"] = {
        key: int(key == "linear")
        for key in raw["coverage_policy"]["operation_definition_counts"]
    }
    raw["coverage_policy"]["profile_minimum_generated_definitions"] = {
        "core": 1
    }
    manifest_path = source_root / "manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(raw))
    free = 16_800_000_000
    alloc = 15_000_000_000
    payload = {
        "schema_version": PlatformArtifactSchema.GPU_MEMORY_QUOTA_EVIDENCE,
        "device": "cuda:0",
        "device_index": 0,
        "gpu_name": "AMD Radeon RX 9060 XT",
        "gfx_target": "gfx1200",
        "visible_compute_units": 16,
        "l2_cache_bytes": 4_194_304,
        "torch_version": "2.14.0+rocm10.1.0",
        "hip_version": "7.16.26385",
        "collected_at": datetime(2026, 11, 19, tzinfo=UTC),
        "runtime_free_bytes": free,
        "runtime_total_bytes": 17_095_983_104,
        "environment_quota_bytes": None,
        "stable_allocatable_bytes": alloc,
        "harness_reserve_bytes": 0,
        "safety_percent": 85,
        "usable_budget_bytes": derive_usable_budget(
            runtime_free_bytes=free,
            environment_quota_bytes=None,
            stable_allocatable_bytes=alloc,
            harness_reserve_bytes=0,
        ),
        "capacity_probe_digest": "0" * 64,
    }
    payload["capacity_probe_digest"] = capacity_probe_digest(
        GPUMemoryQuotaEvidence.model_construct(
            _fields_set=set(payload), **payload
        )
    )
    capacity = GPUMemoryQuotaEvidence.model_validate(payload)
    target_desc = load_target_descriptor(
        ROOT
        / "problems/LLM_CORE/targets/configurations/rx9060xt/standard-16gb.yaml"
    )
    view_dir = generate_corpus(
        manifest_path,
        tmp_path / "rx9060xt-view",
        target=target_desc,
        profiles=(CorpusProfile.CORE,),
        capacity_evidence=capacity,
        require_complete_profile=True,
    )
    loaded = load_materialized_target_view(
        manifest_path, view_dir / "target-view-manifest.yaml"
    )
    assert (
        loaded.view.hardware_context.configuration.visible_compute_units == 32
    )
    assert (
        loaded.view.hardware_context.configuration.visible_memory_bytes
        == 17_095_983_104
    )
    assert loaded.view.hardware_context.observation.visible_compute_units == 16


def _patch_fake_workers(
    monkeypatch,
    definition_name: str,
    *,
    fail_first_smoke_infra: bool = False,
):
    monkeypatch.setattr(
        torch.cuda,
        "_lazy_init",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("GPU init forbidden")
        ),
    )
    mapper = SimpleNamespace(toolchain_identity={"fixture": "synthetic-mapper"})
    monkeypatch.setattr(target_coverage, "OrojenesisRunner", lambda _: mapper)
    monkeypatch.setattr(c2_c3_campaign, "OrojenesisRunner", lambda _: mapper)
    calls: list[tuple[str, str]] = []

    def stage(request, **kwargs):
        calls.append((request.workload_uuid, request.ir_path.value))
        atomic_write_json_value(
            kwargs["observation_path"],
            StageObservation(
                records=[
                    {
                        "stage": "graph_extraction",
                        "wall_seconds": 0.02,
                        "host_process_high_water_bytes": 128 * 1024**2,
                        "device_peak_bytes": None,
                    }
                ]
            ).model_dump(mode="json"),
        )
        if fail_first_smoke_infra and len(calls) == 1:
            return SolarStageAuditOutcome(
                status=SolarReadinessStatus.FAILED,
                analysis_id=request.workload_uuid,
                ir_path=request.ir_path,
                failure_stage=SolarStage.OUTER_BRIDGE,
                reason_code="worker_no_response",
            )
        if request.ir_path.value == "make_fx_aten":
            return SolarStageAuditOutcome(
                status=SolarReadinessStatus.FAILED,
                analysis_id=request.workload_uuid,
                ir_path=request.ir_path,
                failure_stage=SolarStage.IR_CONVERSION,
                reason_code="unsupported_operation",
                message="synthetic unsupported op",
            )
        stages = tuple(
            {
                "stage": s,
                "status": "passed",
                "artifact": {"path": p, "sha256": "a" * 64},
            }
            for s, p in readiness_stage_artifacts(request.ir_path).items()
        )
        return SolarStageAuditOutcome(
            status=SolarReadinessStatus.READY,
            analysis_id=f"{definition_name}:{request.workload_uuid}",
            output_dir=request.output_dir,
            ir_path=request.ir_path,
            architecture_sha256="a" * 64,
            stages=stages,
        )

    def formal(request, **kwargs):
        atomic_write_json_value(
            kwargs["observation_path"],
            StageObservation(
                records=[
                    {
                        "stage": "mapper",
                        "wall_seconds": 0.05,
                        "host_process_high_water_bytes": 256 * 1024**2,
                        "device_peak_bytes": None,
                    }
                ]
            ).model_dump(mode="json"),
        )
        return SolarAnalysisOutcome(
            status=SolarAnalysisStatus.FAILED,
            analysis_id=f"{definition_name}:{request.workload_uuid}",
            ir_path=request.ir_path,
            stage=SolarStage.FORMAL_ANALYSIS,
            reason_code="formal_bound_unproved",
        )

    monkeypatch.setattr(target_coverage, "run_solar_stage_worker", stage)
    monkeypatch.setattr(target_coverage, "run_solar_worker", formal)
    monkeypatch.setattr(
        coverage_evidence, "verify_readiness_receipt", lambda *a: None
    )
    monkeypatch.setattr(
        target_coverage, "verification_dimensions", lambda *a: {"fixture": True}
    )
    return calls


def test_smoke_slice_preserves_exact_resume_identity_for_full_audit(
    tmp_path: Path, monkeypatch
) -> None:
    target = _build_synthetic_target_view(tmp_path)
    def_name = next(iter(target.definitions.values())).name
    calls = _patch_fake_workers(monkeypatch, def_name)
    coverage_dir = tmp_path / "coverage"
    mapper_home = tmp_path / "unused-mapper"
    options = c2_c3_campaign.clamped_coverage_options(mapper_home, "cuda:0")

    smoke_report = c2_c3_campaign._execute_smoke_validated(
        target, coverage_dir, options
    )
    assert len(calls) == 2
    assert len(smoke_report.rows) == 18
    assert all(r.slot == "smoke" and r.attempts for r in smoke_report.rows[:2])
    assert [r.status for r in smoke_report.rows[:2]] == [
        CoverageStatus.UNSUPPORTED,
        CoverageStatus.UNPROVED,
    ]
    assert all(
        not r.attempts and r.status is CoverageStatus.NOT_REACHED
        for r in smoke_report.rows[2:]
    )
    assert smoke_report.matrix_complete is False
    assert smoke_report.complete is False

    # Production _audit_validated_target receives UNCLAMPED CoverageOptions and
    # must resume seamlessly from row 2 without re-running the 2 smoke rows.
    resumed = target_coverage._audit_validated_target(
        target,
        coverage_dir,
        CoverageOptions(device="cuda:0", orojenesis_home=mapper_home),
        resume=True,
    )
    assert len(calls) == 18
    assert resumed.identity == smoke_report.identity
    assert resumed.identity_sha256 == smoke_report.identity_sha256
    assert resumed.matrix_complete is True
    assert resumed.costs_observed is True
    assert resumed.complete is True

    summary = c2_c3_campaign.build_smoke_summary(
        c2_c3_campaign.CampaignPaths(
            repo=ROOT, outputs=tmp_path, freeze=tmp_path / "freeze.json"
        ),
        smoke_report,
        options,
    )
    assert summary["smoke_rows_count"] == 2
    assert summary["deferred_rows_count"] == 16
    assert summary["smoke_status_counts"] == {"unproved": 1, "unsupported": 1}
    assert "readiness:graph_extraction" in summary["stage_cost_summary"]
    assert "formal:mapper" in summary["stage_cost_summary"]
    assert (
        summary["make_fx_real_retry_observability"][
            "observable_in_current_artifacts"
        ]
        is False
    )


def test_smoke_infrastructure_failure_stops_second_smoke_row(
    tmp_path: Path, monkeypatch
) -> None:
    target = _build_synthetic_target_view(tmp_path)
    def_name = next(iter(target.definitions.values())).name
    calls = _patch_fake_workers(
        monkeypatch, def_name, fail_first_smoke_infra=True
    )
    coverage_dir = tmp_path / "coverage-fail"
    options = c2_c3_campaign.clamped_coverage_options(
        tmp_path / "unused-mapper", "cuda:0"
    )
    report = c2_c3_campaign._execute_smoke_validated(
        target, coverage_dir, options
    )
    assert len(calls) == 1
    assert report.rows[0].status is CoverageStatus.INFRASTRUCTURE_ERROR
    assert report.rows[1].status is CoverageStatus.NOT_REACHED
    assert "blocks expansion" in report.rows[1].reason
    assert not report.complete


def test_host_and_campaign_enums_cover_expected_phases() -> None:
    assert {a.value for a in c2_c3_host.HostAction} == {
        "prepare",
        "readiness",
        "b2-representatives",
        "c2",
        "c3-smoke",
        "c3-resume",
        "audit",
    }
    assert {p.value for p in c2_c3_campaign.CampaignPhase} == {
        "readiness",
        "admit-gpu",
        "b2-representatives",
        "c2",
        "c3-smoke",
        "c3-resume",
        "audit",
    }
