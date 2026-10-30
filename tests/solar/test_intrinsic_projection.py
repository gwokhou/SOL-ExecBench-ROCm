"""Reusable intrinsic counts stay independent of profile and capacity choices."""

from dataclasses import replace
from pathlib import Path

import pytest

from scripts.internal.solar.base01_oracles import f_expectations
from scripts.internal.solar.base01_records import PROFILE
from scripts.internal.solar.base01_samples import get_sample
from solar.analysis.graph_analyzer import IRGraphAnalyzer
from solar.analysis.graph_context import AnalysisJob
from solar.artifacts import sha256_file
from solar.ir.contracts import IRPath
from solar.ir.registry import ir_backend
from solar.pipeline.readiness import (
    ConversionReadinessRequest,
    audit_conversion,
)
from solar.rocm.architecture import ArchitectureProfile


@pytest.fixture(params=list(IRPath), ids=lambda item: item.value)
def prepare(request, tmp_path):
    path = request.param

    def build(sample_id):
        sample = get_sample(sample_id)
        root = tmp_path / sample_id
        conversion = sample.request(path, sha256_file(Path(__file__)))
        result = audit_conversion(
            ConversionReadinessRequest(
                conversion=conversion,
                architecture=PROFILE,
                output_dir=root / "conversion",
                require_verified_audit=False,
            )
        )
        assert result.ready, result.to_dict()
        graph = root / "conversion" / path.graph_filename
        # An adjacent candidate must never silently replace the selected graph.
        graph.with_name(f"{graph.stem}_reordered.yaml").write_text(
            "invalid graph"
        )
        analyzer = IRGraphAnalyzer(validator=ir_backend(path.ir_kind).validate)
        job = AnalysisJob(
            graph_path=graph,
            output_dir=root / "analysis",
            precision="fp32",
            copy_graph=False,
            strict=True,
            architecture=PROFILE,
            orojenesis_runner=None,
            require_orojenesis=False,
        )
        intrinsic = analyzer.prepare_intrinsic(job)
        assert intrinsic is not None
        assert intrinsic.prepared.profile is None
        return analyzer, intrinsic, job

    return build


@pytest.mark.parametrize("sample_id", ["F01-D", "F05-D"])
def test_reprojection_reuses_counts_and_reprices_without_reloading(
    prepare, sample_id, monkeypatch, tmp_path
):
    analyzer, intrinsic, job = prepare(sample_id)
    profile = ArchitectureProfile.load(PROFILE)
    full = analyzer.analyze_graph(
        job.graph_path,
        tmp_path / "full",
        strict=True,
        precision=job.precision,
        architecture=profile,
        copy_graph=False,
    )
    assert full is not None

    def forbidden(*args, **kwargs):
        pytest.fail("reprojection must not reload or recount the graph")

    monkeypatch.setattr(analyzer, "_load_graph", forbidden)
    monkeypatch.setattr(analyzer, "_compute_layer", forbidden)
    for name, scenario in (
        ("same", profile),
        (
            "double",
            replace(
                profile,
                memory_bandwidth_bytes_per_second=2
                * profile.memory_bandwidth_bytes_per_second,
                resource_limits={
                    r: {m: 2 * rate for m, rate in modes.items()}
                    for r, modes in profile.resource_limits.items()
                },
            ),
        ),
    ):
        observed = analyzer.project_intrinsic(
            intrinsic,
            output_dir=tmp_path / name,
            architecture=scenario,
        )
        for key, expected in f_expectations()[sample_id].items():
            assert observed["total"][key] == expected
        if name == "same":
            assert observed == full
        else:
            assert observed["total"]["lower_bound_seconds"] == pytest.approx(
                full["total"]["lower_bound_seconds"] / 2
            )


def test_capacity_change_reenters_evidence_analysis(
    prepare, monkeypatch, tmp_path
):
    analyzer, intrinsic, _ = prepare("F01-D")
    profile = ArchitectureProfile.load(PROFILE)
    capacities = []
    original = analyzer._run_formal_analysis

    def record(prepared, *args, **kwargs):
        capacities.append(prepared.profile.last_level_cache_bytes)
        return original(prepared, *args, **kwargs)

    monkeypatch.setattr(analyzer, "_run_formal_analysis", record)
    for capacity in (64, 4096):
        observed = analyzer.project_intrinsic(
            intrinsic,
            output_dir=tmp_path / str(capacity),
            architecture=replace(profile, last_level_cache_bytes=capacity),
        )
        assert observed["metadata"]["orojenesis"]["status"] != "complete"
    assert capacities == [64, 4096]


def test_missing_new_resource_mode_does_not_reuse_old_pricing(
    prepare, tmp_path
):
    analyzer, intrinsic, _ = prepare("F01-D")
    profile = ArchitectureProfile.load(PROFILE)
    changed = replace(
        profile, resource_limits={**profile.resource_limits, "mfma": {}}
    )
    with pytest.raises(ValueError, match="Resource mode"):
        analyzer.project_intrinsic(
            intrinsic,
            output_dir=tmp_path / "missing",
            architecture=changed,
        )


def test_snapshot_retains_loaded_digest_when_source_file_changes(
    prepare, tmp_path
):
    analyzer, intrinsic, _ = prepare("F01-D")
    digest = intrinsic.prepared.source_sha256
    intrinsic.prepared.source.write_text("changed after preparing counts")
    observed = analyzer.project_intrinsic(
        intrinsic,
        output_dir=tmp_path / "snapshot",
        architecture=ArchitectureProfile.load(PROFILE),
    )
    assert observed["metadata"]["source_graph_sha256"] == digest
    assert digest != sha256_file(intrinsic.prepared.source)
