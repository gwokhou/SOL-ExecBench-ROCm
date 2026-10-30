"""M03/F03 capacity selection using declared synthetic mapper receipts.

The frozen curve tests composition, coverage and invalidation. It does not
establish a real mapper lower bound or device qualification.
"""

from pathlib import Path
from typing import cast

import pytest
import yaml

from scripts.internal.solar.base01_records import PROFILE
from scripts.internal.solar.base01_samples import get_sample
from solar.analysis.orojenesis.multi_einsum import multi_einsum_problem
from solar.analysis.views import (
    AnalysisView,
    ViewAvailability,
    read_analysis_views,
)
from solar.api import AnalysisRequest, AnalysisResult, analyze
from solar.artifacts import sha256_file, stable_json_checksum
from solar.artifacts.document import ArtifactDocument
from solar.artifacts.names import (
    CONVERSION_ATTESTATION_FILENAME,
    MANIFEST_FILENAME,
    REUSE_FILENAME,
)
from solar.ir.contracts import IRPath
from solar.pipeline.reuse_artifacts import load_projection
from solar.reuse import ReuseSession
from solar.reuse_contracts import ProjectionRequest, VerifiedProgramRequest
from solar.rocm.architecture import ArchitectureProfile


class FixtureMapper:
    """Test-only fixed curve over the actual F03 chain descriptor."""

    def __init__(self, home, *, timeout_seconds):
        self.mode = Path(home).name
        self.timeout_seconds = timeout_seconds
        self.toolchain_identity = {
            "verification_mode": "fixture_only",
            "fixture_id": self.mode,
        }

    def run_multi_chain(self, layers, output_dir, *, word_bits):
        assert word_bits == 32
        problem = multi_einsum_problem(layers)
        assert [
            (p["m"], p["k"], p["n"]) for p in problem["chain"]["layers"]
        ] == [(3, 5, 4), (3, 4, 2)]
        curve = [
            {"buffer_bytes": 64, "dram_bytes": 392.0},
            {"buffer_bytes": 4096, "dram_bytes": 196.0},
        ]
        if self.mode == "missing-point":
            curve = curve[1:]
        output_dir.mkdir(parents=True)
        raw = output_dir / "fixture-curve.yaml"
        raw.write_text(yaml.safe_dump({"fixture_only": True, "curve": curve}))
        digest = "0" * 64 if self.mode == "corrupt-proof" else sha256_file(raw)
        return {
            "word_bits": word_bits,
            "problem": problem,
            "curve": curve,
            "evidence_files": {"curve": {"path": raw.name, "sha256": digest}},
        }


@pytest.fixture(
    scope="module", params=list(IRPath), ids=lambda path: path.value
)
def source(request, tmp_path_factory):
    root = tmp_path_factory.mktemp("capacity-reuse") / "source"
    conversion = get_sample("F03-D").request(
        request.param, sha256_file(Path(__file__))
    )
    result = analyze(
        AnalysisRequest(
            conversion=conversion,
            architecture=PROFILE,
            output_dir=root,
            precision="fp32",
            require_verified_audit=False,
        )
    )
    assert isinstance(result, AnalysisResult), result
    receipt = yaml.safe_load(
        (root / CONVERSION_ATTESTATION_FILENAME).read_text()
    )
    return VerifiedProgramRequest(
        directory=root,
        manifest_sha256=sha256_file(root / MANIFEST_FILENAME),
        reference_sha256=conversion.reference_sha256,
        execution_sha256=stable_json_checksum(
            receipt["predicate"]["execution"]
        ),
        ir_path=request.param,
        verification=conversion.verification,
    )


def projection(tmp_path, capacity, mode):
    profile = ArchitectureProfile.load(PROFILE).to_dict()
    profile["last_level_cache_bytes"] = capacity
    profile["memory_hierarchy"] = [
        {"name": "cache", "scope": "device", "capacity_bytes": capacity}
    ]
    return ProjectionRequest(
        architecture=profile,
        output_dir=tmp_path / f"{capacity}-{mode}",
        orojenesis_home=tmp_path / mode,
    )


def test_m03_f03_complete_capacity_curve_and_evidence_bindings(
    source, tmp_path, monkeypatch
):
    monkeypatch.setattr(
        "solar.pipeline.reuse_projection.OrojenesisRunner", FixtureMapper
    )
    session = ReuseSession(source)
    results = [
        session.project(projection(tmp_path, size, "complete"))
        for size in (64, 4096)
    ]
    counts = []
    for result in results:
        document = ArtifactDocument(result.output_dir, result.analysis)
        metadata = document.require_mapping("metadata")
        counts.append(document.require_mapping("total")["macs"])
        view = read_analysis_views(metadata["views"])[AnalysisView.CACHE_AWARE]
        assert view.status is ViewAvailability.AVAILABLE
        assert not result.record.publication_eligible
        configured = result.record.identity_inputs["projection"]
        assert isinstance(configured, dict)
        mapper = cast(dict[str, object], configured)["mapper"]
        assert isinstance(mapper, dict)
        assert (
            cast(dict[str, object], mapper)["verification_mode"]
            == "fixture_only"
        )
        assert any(
            name.startswith("orojenesis/") for name in result.record.artifacts
        )
        assert (
            load_projection(
                result.output_dir,
                record_sha256=sha256_file(result.output_dir / REUSE_FILENAME),
                projection_sha256=result.record.projection_sha256,
            )
            == result
        )
    views = [
        read_analysis_views(
            ArtifactDocument(r.output_dir, r.analysis).require_mapping(
                "metadata"
            )["views"]
        )[AnalysisView.CACHE_AWARE]
        for r in results
    ]
    assert [v.io_bytes for v in views] == [392, 196]
    assert counts == [84, 84]
    assert views[0].seconds is not None and views[1].seconds is not None
    assert views[0].seconds >= views[1].seconds
    assert (
        results[0].record.intrinsic_sha256 == results[1].record.intrinsic_sha256
    )
    assert (
        results[0].record.projection_sha256
        != results[1].record.projection_sha256
    )


@pytest.mark.parametrize("mode", ["missing-point", "corrupt-proof"])
def test_missing_capacity_or_invalid_proof_keeps_cache_view_unavailable(
    source, tmp_path, monkeypatch, mode
):
    monkeypatch.setattr(
        "solar.pipeline.reuse_projection.OrojenesisRunner", FixtureMapper
    )
    result = ReuseSession(source).project(projection(tmp_path, 64, mode))
    metadata = ArtifactDocument(
        result.output_dir, result.analysis
    ).require_mapping("metadata")
    view = read_analysis_views(metadata["views"])[AnalysisView.CACHE_AWARE]
    assert view.status is ViewAvailability.UNAVAILABLE
    assert view.seconds is None and view.io_bytes is None
    assert not result.record.publication_eligible


def test_mapper_identity_change_changes_projection_identity(
    source, tmp_path, monkeypatch
):
    monkeypatch.setattr(
        "solar.pipeline.reuse_projection.OrojenesisRunner", FixtureMapper
    )
    session = ReuseSession(source)
    first = session.project(projection(tmp_path, 64, "mapper-a"))
    second = session.project(projection(tmp_path, 64, "mapper-b"))
    assert first.record.intrinsic_sha256 == second.record.intrinsic_sha256
    assert first.record.projection_sha256 != second.record.projection_sha256


def test_scan_capacity_recomputes_curves_and_reports_missing_coverage(
    source, tmp_path, monkeypatch
):
    from solar.analysis.scan_contracts import AxisKind, PointStatus, ScanAxis
    from solar.analysis.scan_reporting import report_scan
    from solar.explore import scan_profiles
    from solar.reuse_contracts import ReuseJob
    from solar.scan_contracts import ScanRequest

    monkeypatch.setattr(
        "solar.pipeline.reuse_projection.OrojenesisRunner", FixtureMapper
    )
    for mode in ("complete", "missing-point"):
        template = projection(tmp_path, 64, mode)
        request = ScanRequest(
            job=ReuseJob(source=source, projection=template),
            axes=(
                ScanAxis(
                    kind=AxisKind.CAPACITY_BYTES,
                    memory_level="cache",
                    values=(64, 4096),
                ),
            ),
            view=AnalysisView.CACHE_AWARE,
        )
        result = scan_profiles(request)
        assert len(result.points) == 2
        assert result.points[1].estimate is not None, result.model_dump_json()
        assert result.points[1].estimate.io_bytes == 196
        assert result.points[0].estimate is not None, result.model_dump_json()
        assert result.points[1].status is PointStatus.ANALYZED
        if mode == "complete":
            assert result.points[0].estimate.io_bytes == 392
            assert result.points[0].estimate.seconds is not None
            assert result.points[1].estimate.seconds is not None
            assert (
                result.points[0].estimate.seconds
                >= result.points[1].estimate.seconds
            )
        else:
            assert result.points[0].status is PointStatus.UNAVAILABLE
            assert result.points[0].estimate.seconds is None
        report = report_scan(request, result)
        assert report["planned_points"] == 2
        assert report["available_points"] == (2 if mode == "complete" else 1)
        assert report["publication_eligible"] is False
