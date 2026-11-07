"""Public fixed-instance reuse and its atomic, content-bound outputs."""

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from scripts.internal.solar.base01_oracles import f_expectations
from scripts.internal.solar.base01_records import PROFILE
from scripts.internal.solar.base01_samples import get_sample
from solar.analysis.access_regions import AccessBudget
from solar.analysis.exploration_contracts import (
    InverseRequest,
    ProjectionReference,
)
from solar.analysis.formal_analysis import FormalBoundAnalyzer
from solar.analysis.graph_analyzer import IRGraphAnalyzer
from solar.analysis.graph_loading import GraphLoader
from solar.analysis.identity import RuleGroup, analysis_rules_identity
from solar.analysis.views import AnalysisView, read_analysis_views
from solar.api import AnalysisRequest, AnalysisResult, analyze
from solar.artifacts import sha256_file, stable_json_checksum
from solar.artifacts.document import ArtifactDocument
from solar.artifacts.names import (
    ANALYSIS_FILENAME,
    CONVERSION_ATTESTATION_FILENAME,
    MANIFEST_FILENAME,
    REUSE_FILENAME,
)
from solar.explore import query_resources
from solar.ir.contracts import IRPath
from solar.pipeline.reuse_artifacts import load_projection
from solar.reuse import ReuseSession
from solar.reuse_contracts import (
    ProfileScenario,
    ProjectionRequest,
    ReuseOutputExistsError,
    ReuseRulesChangedError,
    VerifiedProgramRequest,
)
from solar.rocm.architecture import ArchitectureProfile


@pytest.fixture(
    scope="module",
    params=[(sample, path) for sample in ("F01-D", "F05-D") for path in IRPath],
    ids=lambda item: f"{item[0]}-{item[1].value}",
)
def source(request, tmp_path_factory):
    sample_id, path = request.param
    root = tmp_path_factory.mktemp("reuse") / "source"
    conversion = get_sample(sample_id).request(
        path, sha256_file(Path(__file__))
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
    identity = VerifiedProgramRequest(
        directory=root,
        manifest_sha256=sha256_file(root / MANIFEST_FILENAME),
        reference_sha256=conversion.reference_sha256,
        ir_path=path,
        verification=conversion.verification,
        execution_sha256=stable_json_checksum(
            receipt["predicate"]["execution"]
        ),
    )
    return (
        sample_id,
        identity,
        yaml.safe_load((root / ANALYSIS_FILENAME).read_text()),
    )


def test_full_and_public_reuse_match_without_execution_or_reloading(
    source, tmp_path, monkeypatch
):
    sample_id, identity, full = source

    def forbidden(*args, **kwargs):
        pytest.fail(
            "reuse must not execute a reference, extract or reopen graph files"
        )

    monkeypatch.setattr(
        "solar.pipeline.stages.extract_request_graph", forbidden
    )
    monkeypatch.setattr("solar.verification.verify._run_cases", forbidden)
    monkeypatch.setattr("solar.verification.verify._load_module", forbidden)
    monkeypatch.setattr(GraphLoader, "_load_graph", forbidden)
    session = ReuseSession(identity)
    monkeypatch.setattr(GraphLoader, "_compute_layer", forbidden)
    result = session.project(
        ProjectionRequest(architecture=PROFILE, output_dir=tmp_path / "same")
    )
    assert result.analysis == full
    assert result.record.scenario is ProfileScenario.IDENTICAL
    assert result.record.sol_score_eligible
    assert not result.record.publication_eligible
    totals = ArtifactDocument(
        path=result.output_dir, data=result.analysis
    ).require_mapping("total")
    for name, expected in f_expectations()[sample_id].items():
        assert totals[name] == expected
    restored = load_projection(
        result.output_dir,
        record_sha256=sha256_file(result.output_dir / REUSE_FILENAME),
        projection_sha256=result.record.projection_sha256,
    )
    assert restored == result


def test_rate_change_reuses_counts_and_drops_inherited_audit(
    source, tmp_path, monkeypatch
):
    _, identity, full = source
    session = ReuseSession(identity)
    profile = ArchitectureProfile.load(PROFILE).to_dict()
    profile["memory_bandwidth_bytes_per_second"] *= 2
    for modes in profile["resource_limits"].values():
        for mode in modes:
            modes[mode] *= 2
    # A superficially valid old audit must not grant the new scenario authority.
    profile["audit_evidence"] = ArchitectureProfile.load(
        "RX_9060_XT"
    ).audit_evidence

    def forbidden(*args, **kwargs):
        pytest.fail("rate projection must not recount intrinsic work")

    monkeypatch.setattr(GraphLoader, "_compute_layer", forbidden)
    result = session.project(
        ProjectionRequest(architecture=profile, output_dir=tmp_path / "double")
    )
    assert result.record.scenario is ProfileScenario.HYPOTHETICAL
    assert not result.record.sol_score_eligible
    assert not result.record.publication_eligible
    assert not result.record.profile_audit_verified
    metadata = ArtifactDocument(
        path=result.output_dir, data=result.analysis
    ).require_mapping("metadata")
    architecture = ArtifactDocument(
        path=result.output_dir, data=metadata
    ).require_mapping("architecture")
    assert architecture["audit_evidence"] == {
        "status": "unavailable",
        "reason_code": "hypothetical_profile",
    }
    views = read_analysis_views(metadata["views"])
    for view, previous in full["metadata"]["views"].items():
        observed = views[view]
        assert observed.status == previous["status"]
        if previous["seconds"] is not None:
            assert observed.seconds == pytest.approx(previous["seconds"] / 2)


def test_capacity_and_timeout_enter_identity_and_recompute_evidence(
    source, tmp_path, monkeypatch
):
    _, identity, _ = source
    session = ReuseSession(identity)
    profile = ArchitectureProfile.load(PROFILE).to_dict()
    seen = []
    original = FormalBoundAnalyzer._run_formal_analysis

    def record(self, prepared, *args, **kwargs):
        seen.append(prepared.profile.last_level_cache_bytes)
        return original(self, prepared, *args, **kwargs)

    monkeypatch.setattr(FormalBoundAnalyzer, "_run_formal_analysis", record)
    results = []
    for capacity, timeout in ((64, 5), (4096, 5), (4096, 6)):
        profile["last_level_cache_bytes"] = capacity
        results.append(
            session.project(
                ProjectionRequest(
                    architecture=deepcopy(profile),
                    output_dir=tmp_path / f"c{capacity}-t{timeout}",
                    mapper_timeout_seconds=timeout,
                )
            )
        )
    assert seen == [64, 4096, 4096]
    assert len({r.record.intrinsic_sha256 for r in results}) == 1
    assert len({r.record.projection_sha256 for r in results}) == 3


def test_rule_drift_requires_repreparation_before_writing(
    source, tmp_path, monkeypatch
):
    _, identity, _ = source
    session = ReuseSession(identity)
    rules = analysis_rules_identity()
    rules[RuleGroup.COUNTS] = "0" * 64
    monkeypatch.setattr("solar.reuse.analysis_rules_identity", lambda: rules)
    output = tmp_path / "changed"
    with pytest.raises(ReuseRulesChangedError):
        session.project(
            ProjectionRequest(architecture=PROFILE, output_dir=output)
        )
    assert not output.exists()


def test_access_budget_is_part_of_intrinsic_identity(source):
    _, identity, _ = source
    small = ReuseSession(identity, access_budget=AccessBudget(seconds=1))
    large = ReuseSession(identity, access_budget=AccessBudget(seconds=2))
    assert small.intrinsic_sha256 != large.intrinsic_sha256


def test_resource_mode_changes_are_reclassified_before_pricing(
    source, tmp_path
):
    sample_id, identity, full = source
    session = ReuseSession(identity)
    profile = ArchitectureProfile.load(PROFILE).to_dict()
    # This is a valid profile with another contraction mode, not a malformed
    # empty resource group. F01 needs fp32 MFMA; F05 only converts/adds values.
    profile["resource_limits"]["mfma"] = {"fp64": 1e9}
    profile["calibration_exempt_modes"]["mfma"] = {"fp64": "CPU fixture only"}
    request = ProjectionRequest(
        architecture=profile, output_dir=tmp_path / "modes"
    )
    if sample_id == "F01-D":
        with pytest.raises(ValueError, match="Resource mode"):
            session.project(request)
        assert not request.output_dir.exists()
    else:
        result = session.project(request)
        assert (
            ArtifactDocument(
                path=result.output_dir, data=result.analysis
            ).require_mapping("total")
            == full["total"]
        )
        assert not result.record.publication_eligible


def test_returned_results_cannot_mutate_prepared_counts_or_identity(
    source, tmp_path
):
    _, identity, full = source
    session = ReuseSession(identity)
    first = session.project(
        ProjectionRequest(architecture=PROFILE, output_dir=tmp_path / "first")
    )
    original_key = session.intrinsic_sha256

    # Arbitrary user edits to an output must not poison the next projection.
    def erase(value):
        if isinstance(value, dict):
            for item in value.values():
                erase(item)
            value.clear()
        elif isinstance(value, list):
            for item in value:
                erase(item)
            value.clear()

    erase(first.analysis)
    first.record.identity_inputs.clear()
    first.record.rules.clear()
    second = session.project(
        ProjectionRequest(architecture=PROFILE, output_dir=tmp_path / "second")
    )
    assert session.intrinsic_sha256 == original_key
    assert second.analysis == full


def test_existing_output_and_failed_output_budget_leave_no_partial_result(
    source, tmp_path
):
    _, identity, _ = source
    session = ReuseSession(identity)
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "retain.txt"
    marker.write_text("previous attempt")
    with pytest.raises(ReuseOutputExistsError):
        session.project(
            ProjectionRequest(architecture=PROFILE, output_dir=output)
        )
    assert marker.read_text() == "previous attempt"
    with pytest.raises(ValueError, match="byte budget"):
        session.project(
            ProjectionRequest(
                architecture=PROFILE,
                output_dir=tmp_path / "too-small",
                output_bytes=1,
            )
        )
    assert sorted(p.name for p in tmp_path.iterdir()) == ["existing"]


@pytest.mark.parametrize(
    "damage", ["missing", "changed", "extra", "record-identity"]
)
def test_partial_or_changed_projection_cannot_be_resumed(
    source, tmp_path, damage
):
    _, identity, _ = source
    result = ReuseSession(identity).project(
        ProjectionRequest(architecture=PROFILE, output_dir=tmp_path / "result")
    )
    record_sha256 = sha256_file(result.output_dir / REUSE_FILENAME)
    if damage == "missing":
        (result.output_dir / ANALYSIS_FILENAME).unlink()
    elif damage == "changed":
        (result.output_dir / ANALYSIS_FILENAME).write_text("invalid")
    elif damage == "extra":
        (result.output_dir / "unfinished.tmp").touch()
    else:
        record_sha256 = "0" * 64
    with pytest.raises(ValueError, match=r"inventory|SHA-256"):
        load_projection(
            result.output_dir,
            record_sha256=record_sha256,
            projection_sha256=result.record.projection_sha256,
        )


@pytest.mark.parametrize(
    "field", ["bandwidth", "peak", "resource", "hierarchy"]
)
@pytest.mark.parametrize(
    "value", [float("nan"), float("inf")], ids=["nan", "inf"]
)
def test_nonfinite_profile_rates_rejected_before_pricing(field, value):
    profile = ArchitectureProfile.load(PROFILE).to_dict()
    if field == "bandwidth":
        profile["memory_bandwidth_bytes_per_second"] = value
    elif field == "peak":
        profile["peak_ops_per_second"]["fp32"] = value
    elif field == "resource":
        profile["resource_limits"]["mfma"]["fp32"] = value
    else:
        profile["memory_hierarchy"] = [
            {
                "name": "dram",
                "scope": "device",
                "capacity_bytes": None,
                "bandwidth_bytes_per_second": value,
            }
        ]
    with pytest.raises(ValueError, match="positive"):
        ArchitectureProfile.load(profile)


def test_pinned_public_inverse_matches_owned_forward_analysis(source, tmp_path):
    _, identity, _ = source
    result = ReuseSession(identity).project(
        ProjectionRequest(
            architecture=PROFILE, output_dir=tmp_path / "inverse-source"
        )
    )
    document = ArtifactDocument(path=result.output_dir, data=result.analysis)
    metadata = document.require_mapping("metadata")
    target = read_analysis_views(metadata["views"])[
        AnalysisView.IDEAL_FUSED
    ].seconds
    assert target is not None
    total = ArtifactDocument(
        path=result.output_dir, data=document.require_mapping("total")
    )
    resource_work = total.require_mapping("resource_work")
    query = InverseRequest(
        source=ProjectionReference(
            directory=result.output_dir,
            record_sha256=sha256_file(result.output_dir / REUSE_FILENAME),
            projection_sha256=result.record.projection_sha256,
        ),
        view=AnalysisView.IDEAL_FUSED,
        target_seconds=target,
        scale_resources=(*tuple(sorted(resource_work)), "dram"),
    )
    answer = query_resources(query)
    assert answer.required_group_scale == pytest.approx(1.0)
    assert answer.forward_seconds_at_group_scale == pytest.approx(target)
    assert answer.projection_sha256 == result.record.projection_sha256
    assert not answer.performance_authority
    (result.output_dir / ANALYSIS_FILENAME).write_text("corrupt")
    with pytest.raises(ValueError, match=r"SHA-256|digest|hash"):
        query_resources(query)


def scan_request(identity, output, *, budget=None, resume=None):
    from solar.analysis.scan_contracts import AxisKind, ScanAxis, ScanBudget
    from solar.reuse_contracts import ReuseJob
    from solar.scan_contracts import ScanRequest

    return ScanRequest(
        job=ReuseJob(
            source=identity,
            projection=ProjectionRequest(
                architecture=PROFILE, output_dir=output
            ),
        ),
        axes=(
            ScanAxis(
                kind=AxisKind.RESOURCE_SCALE,
                resources=tuple(
                    ArchitectureProfile.load(PROFILE).resource_limits
                ),
                values=(1, 2),
            ),
            ScanAxis(kind=AxisKind.BANDWIDTH_SCALE, values=(1, 2)),
        ),
        view=AnalysisView.IDEAL_FUSED,
        budget=budget or ScanBudget(),
        resume=resume,
    )


def test_scan_counts_once_and_matches_independent_forward_rates(
    source, tmp_path, monkeypatch
):
    from solar.explore import scan_profiles
    from solar.pipeline.exploration import read_projection

    _, identity, full = source
    counted = []
    original = IRGraphAnalyzer.count_prepared

    def count(self, *args, **kwargs):
        counted.append(True)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(IRGraphAnalyzer, "count_prepared", count)
    result = scan_profiles(scan_request(identity, tmp_path / "grid"))
    assert len(counted) == 1
    assert len(result.points) == len({p.point_id for p in result.points}) == 4
    profile = ArchitectureProfile.load(PROFILE)
    compute = max(
        sum(
            work / profile.resource_rate_for(resource, mode)
            for mode, work in modes.items()
        )
        for resource, modes in full["total"]["resource_work"].items()
    )
    memory = (
        full["total"]["fused_bytes"] / profile.memory_bandwidth_bytes_per_second
    )
    expected = [max(compute / c, memory / b) for c in (1, 2) for b in (1, 2)]
    assert all(point.estimate is not None for point in result.points)
    assert [
        point.estimate.seconds
        for point in result.points
        if point.estimate is not None
    ] == pytest.approx(expected)
    for point in result.points:
        assert point.result is not None
        reused = read_projection(point.result)
        assert not reused.record.sol_score_eligible
        assert not reused.record.publication_eligible
        assert not reused.record.profile_audit_verified
        assert point.views is not None
        assert len(point.views) == 3


def test_complete_scan_resume_reuses_pinned_results_without_projecting(
    source, tmp_path, monkeypatch
):
    from solar.analysis.scan_contracts import SCAN_FILENAME, ScanReference
    from solar.explore import scan_profiles

    _, identity, _ = source
    request = scan_request(identity, tmp_path / "first")
    first = scan_profiles(request)
    before = sha256_file(tmp_path / "first" / SCAN_FILENAME)

    def forbidden(*args, **kwargs):
        pytest.fail("complete pinned scan points must not be projected again")

    monkeypatch.setattr(ReuseSession, "project", forbidden)
    second = scan_profiles(
        scan_request(
            identity,
            tmp_path / "resumed",
            resume=ScanReference(
                directory=tmp_path / "first", record_sha256=before
            ),
        )
    )
    assert first.scan_sha256 == second.scan_sha256
    assert all(point.resumed for point in second.points)
    assert [p.result for p in first.points] == [p.result for p in second.points]
    assert sha256_file(tmp_path / "first" / SCAN_FILENAME) == before


def test_scan_partial_failure_resume_and_fresh_run_agree(
    source, tmp_path, monkeypatch
):
    from solar.analysis.scan_contracts import (
        SCAN_FILENAME,
        PointStatus,
        ScanReference,
    )
    from solar.errors import ToolchainUnavailableError
    from solar.explore import scan_profiles

    _, identity, _ = source
    original = ReuseSession.project
    calls = []

    def fail_second(self, request):
        calls.append(request.output_dir)
        if len(calls) == 2:
            raise ToolchainUnavailableError("controlled missing fixture proof")
        return original(self, request)

    monkeypatch.setattr(ReuseSession, "project", fail_second)
    first = scan_profiles(scan_request(identity, tmp_path / "partial"))
    assert [p.status for p in first.points].count(PointStatus.FAILED) == 1
    assert len(first.points) == 4
    calls.clear()

    def observed(self, request):
        calls.append(request.output_dir)
        return original(self, request)

    monkeypatch.setattr(ReuseSession, "project", observed)
    second = scan_profiles(
        scan_request(
            identity,
            tmp_path / "resumed",
            resume=ScanReference(
                directory=tmp_path / "partial",
                record_sha256=sha256_file(tmp_path / "partial" / SCAN_FILENAME),
            ),
        )
    )
    assert len(calls) == 1
    assert sum(p.resumed for p in second.points) == 3
    fresh = scan_profiles(scan_request(identity, tmp_path / "fresh"))
    assert all(p.result is not None for p in (*second.points, *fresh.points))
    assert [
        p.result.projection_sha256
        for p in second.points
        if p.result is not None
    ] == [
        p.result.projection_sha256 for p in fresh.points if p.result is not None
    ]
    assert [p.estimate for p in second.points] == [
        p.estimate for p in fresh.points
    ]


@pytest.mark.parametrize("drift", ["budget", "profile", "artifact"])
def test_scan_resume_drift_is_rejected(source, tmp_path, drift):
    from solar.analysis.scan_contracts import SCAN_FILENAME, ScanReference
    from solar.explore import scan_profiles

    _, identity, _ = source
    first = scan_profiles(scan_request(identity, tmp_path / "first"))
    request = scan_request(
        identity,
        tmp_path / "resume",
        resume=ScanReference(
            directory=tmp_path / "first",
            record_sha256=sha256_file(tmp_path / "first" / SCAN_FILENAME),
        ),
    )
    if drift == "budget":
        request = request.model_copy(
            update={
                "budget": request.budget.model_copy(
                    update={"point_seconds": request.budget.point_seconds + 1}
                )
            }
        )
    elif drift == "profile":
        profile = ArchitectureProfile.load(PROFILE).to_dict()
        profile["memory_bandwidth_bytes_per_second"] *= 2
        request = request.model_copy(
            update={
                "job": request.job.model_copy(
                    update={
                        "projection": request.job.projection.model_copy(
                            update={"architecture": profile}
                        )
                    }
                )
            }
        )
    else:
        assert first.points[0].result is not None
        (first.points[0].result.directory / ANALYSIS_FILENAME).write_text(
            "corrupt"
        )
    with pytest.raises(
        ValueError, match=r"identity differs|SHA-256|digest|hash"
    ):
        scan_profiles(request)


def test_scan_total_budget_preserves_all_unattempted_points(
    source, tmp_path, monkeypatch
):
    from solar import scan
    from solar.analysis.scan_contracts import PointStatus, ScanBudget
    from solar.explore import scan_profiles

    _, identity, _ = source
    ticks = iter((0.0, 2.0, 2.0))
    monkeypatch.setattr(scan, "monotonic", lambda: next(ticks))
    result = scan_profiles(
        scan_request(
            identity, tmp_path / "budget", budget=ScanBudget(total_seconds=1.0)
        )
    )
    assert len(result.points) == 4
    assert all(
        p.status is PointStatus.NOT_RUN and not p.attempted
        for p in result.points
    )
    assert result.stop_reason == "total_budget_exhausted"


def test_scan_point_timeout_retains_failure_and_continues(
    source, tmp_path, monkeypatch
):
    import signal
    import time

    from solar.analysis.scan_contracts import PointStatus, ScanBudget
    from solar.explore import scan_profiles

    _, identity, _ = source
    original = ReuseSession.project
    first = True

    def delayed(self, request):
        nonlocal first
        if first:
            first = False
            time.sleep(5)
        return original(self, request)

    monkeypatch.setattr(ReuseSession, "project", delayed)
    result = scan_profiles(
        scan_request(
            identity, tmp_path / "timeout", budget=ScanBudget(point_seconds=1.0)
        )
    )
    assert result.points[0].status is PointStatus.TIMEOUT
    assert all(p.status is PointStatus.ANALYZED for p in result.points[1:])
    assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)


def test_scan_missing_mapper_preserves_every_failed_point(source, tmp_path):
    from solar.analysis.scan_contracts import PointStatus
    from solar.explore import scan_profiles

    _, identity, _ = source
    request = scan_request(identity, tmp_path / "missing-mapper")
    request = request.model_copy(
        update={
            "job": request.job.model_copy(
                update={
                    "projection": request.job.projection.model_copy(
                        update={
                            "require_orojenesis": True,
                            "orojenesis_home": tmp_path / "absent-tool",
                        }
                    )
                }
            )
        }
    )
    result = scan_profiles(request)
    assert len(result.points) == 4
    assert all(
        p.status is PointStatus.FAILED and p.reason == "toolchain_unavailable"
        for p in result.points
    )
    assert all(p.result is None for p in result.points)


def test_unexpected_scan_invariant_is_retained_and_propagates(
    source, tmp_path, monkeypatch
):
    from solar.analysis.scan_contracts import (
        SCAN_FILENAME,
        PointStatus,
        ScanReference,
    )
    from solar.explore import scan_profiles
    from solar.pipeline.scan_artifacts import load_scan

    _, identity, _ = source

    def broken(*args, **kwargs):
        raise RuntimeError("controlled invariant violation")

    monkeypatch.setattr(ReuseSession, "project", broken)
    request = scan_request(identity, tmp_path / "invariant")
    with pytest.raises(RuntimeError, match="controlled invariant"):
        scan_profiles(request)
    snapshot = load_scan(
        ScanReference(
            directory=tmp_path / "invariant",
            record_sha256=sha256_file(tmp_path / "invariant" / SCAN_FILENAME),
        )
    )
    assert snapshot.stop_reason == "interrupted"
    assert snapshot.points[0].status is PointStatus.FAILED
    assert snapshot.points[0].reason == "invariant_failure"
    assert all(p.status is PointStatus.NOT_RUN for p in snapshot.points[1:])


def test_scan_budget_after_partial_success_retains_every_remaining_slot(
    source, tmp_path, monkeypatch
):
    from solar import scan
    from solar.analysis.scan_contracts import PointStatus, ScanBudget
    from solar.explore import scan_profiles

    _, identity, _ = source
    now = [0.0]
    original = ReuseSession.project
    monkeypatch.setattr(scan, "monotonic", lambda: now[0])

    def finish_then_exhaust(self, request):
        result = original(self, request)
        now[0] = 2.0
        return result

    monkeypatch.setattr(ReuseSession, "project", finish_then_exhaust)
    result = scan_profiles(
        scan_request(
            identity,
            tmp_path / "partial-budget",
            budget=ScanBudget(total_seconds=1.0),
        )
    )
    assert result.points[0].status is PointStatus.ANALYZED
    assert all(
        p.status is PointStatus.NOT_RUN and not p.attempted
        for p in result.points[1:]
    )
    assert result.stop_reason == "total_budget_exhausted"


@pytest.mark.parametrize("description", ["source", "resource", "memory"])
def test_description_change_preserves_qualification_but_changes_full_identity(
    source, tmp_path, description
):
    _, identity, full = source
    profile = ArchitectureProfile.load(PROFILE).to_dict()
    if description == "source":
        profile["source"] = "Clarified source description"
    elif description == "resource":
        profile["resource_limit_sources"]["valu"] = "Clarified source citation"
    else:
        profile["memory_hierarchy"][0]["source"] = (
            "Clarified memory description"
        )
    result = ReuseSession(identity).project(
        ProjectionRequest(architecture=profile, output_dir=tmp_path / "wording")
    )
    assert result.record.scenario is ProfileScenario.IDENTICAL
    assert result.record.sol_score_eligible
    assert not result.record.publication_eligible
    assert not result.record.profile_audit_verified
    assert result.analysis["total"] == full["total"]
    metadata = ArtifactDocument(
        path=result.output_dir, data=result.analysis
    ).require_mapping("metadata")
    observed = metadata["architecture"]
    assert observed == profile
    assert result.record.requested_architecture_sha256 == stable_json_checksum(
        profile
    )
    assert result.record.architecture_sha256 != stable_json_checksum(
        full["metadata"]["architecture"]
    )
    assert (
        load_projection(
            result.output_dir,
            record_sha256=sha256_file(result.output_dir / REUSE_FILENAME),
            projection_sha256=result.record.projection_sha256,
        )
        == result
    )


@pytest.mark.parametrize("edit", ["description", "bandwidth"])
def test_source_profile_binding_survives_resealed_inventory(
    source, tmp_path, edit
):
    from shutil import copytree

    from solar.reuse_contracts import ReuseArtifactError

    _, identity, _ = source
    directory = tmp_path / "resealed"
    copytree(identity.directory, directory)
    analysis_path = directory / ANALYSIS_FILENAME
    analysis = yaml.safe_load(analysis_path.read_text())
    profile = analysis["metadata"]["architecture"]
    if edit == "description":
        profile["source"] = "Tampered source description"
    else:
        profile["memory_bandwidth_bytes_per_second"] *= 2
    analysis_path.write_text(yaml.safe_dump(analysis))
    manifest_path = directory / MANIFEST_FILENAME
    manifest = yaml.safe_load(manifest_path.read_text())
    for artifact in manifest["artifacts"]:
        if artifact["path"] == ANALYSIS_FILENAME:
            artifact["sha256"] = sha256_file(analysis_path)
    manifest_path.write_text(yaml.safe_dump(manifest))
    request = identity.model_copy(
        update={
            "directory": directory,
            "manifest_sha256": sha256_file(manifest_path),
        }
    )
    with pytest.raises(
        ReuseArtifactError, match="analysis identity or bound differs"
    ):
        ReuseSession(request)
