"""Restricted computation keeps the independent dense and coordinate oracles."""

from dataclasses import replace
from pathlib import Path

import pytest
import torch
import yaml

from scripts.internal.solar.base01_records import PROFILE
from scripts.internal.solar.base01_samples import get_sample
from solar.analysis.demand import restrict_computation
from solar.analysis.demand_contracts import (
    DEMAND_FILENAME,
    ComputationScenario,
    DemandBudget,
    DemandReason,
    DemandRequest,
    DemandStatus,
    NumericalDomain,
)
from solar.analysis.demand_plan import (
    DEMAND_TIME_CHECK_INTERVAL,
    DemandDeclinedError,
    DemandWork,
    checked_items,
    coordinates,
)
from solar.analysis.demand_reporting import demand_report
from solar.analysis.exploration_contracts import (
    InverseRequest,
    InverseStatus,
    ProjectionReference,
)
from solar.analysis.graph_analyzer import IRGraphAnalyzer
from solar.analysis.graph_context import AnalysisJob
from solar.analysis.scan_contracts import (
    AxisKind,
    PointStatus,
    ScanAxis,
    ScanReference,
)
from solar.analysis.views import AnalysisView, ViewAvailability
from solar.api import (
    AnalysisRequest,
    AnalysisResult,
    ConversionReadinessRequest,
    analyze,
    audit_conversion,
)
from solar.artifacts import sha256_file, stable_json_checksum
from solar.artifacts.document import ArtifactDocument
from solar.artifacts.names import (
    CONVERSION_ATTESTATION_FILENAME,
    MANIFEST_FILENAME,
    REUSE_FILENAME,
)
from solar.explore import query_resources, scan_profiles
from solar.ir.contracts import IRPath
from solar.pipeline.reuse_artifacts import load_projection
from solar.reuse import ReuseSession
from solar.reuse_contracts import (
    ProjectionRequest,
    ReuseJob,
    VerifiedProgramRequest,
)
from solar.rocm.architecture import ArchitectureProfile
from solar.scan_contracts import ScanRequest

FINITE = DemandRequest(numerical_domain=NumericalDomain.FINITE)


@pytest.fixture(
    scope="module",
    params=[(sample, path) for sample in ("F06-D", "F06-Q") for path in IRPath],
    ids=lambda p: f"{p[0]}-{p[1].value}",
)
def triangle_source(request, tmp_path_factory):
    sample_id, path = request.param
    root = tmp_path_factory.mktemp("demand-source") / "source"
    sample = get_sample(sample_id)
    conversion = sample.request(path, sha256_file(Path(__file__)))
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
        execution_sha256=stable_json_checksum(
            receipt["predicate"]["execution"]
        ),
        ir_path=path,
        verification=conversion.verification,
    )
    return sample, identity


def test_both_paths_dense_and_restricted_f06_match_coordinate_oracle(
    triangle_source, tmp_path
):
    sample, identity = triangle_source
    result = ReuseSession(identity).project(
        ProjectionRequest(
            architecture=PROFILE,
            output_dir=tmp_path / "restricted",
            restricted=FINITE,
        )
    )
    report = result.restricted
    assert report is not None and report.status is DemandStatus.AVAILABLE
    rows, width = sample.shapes[0]
    columns = sample.shapes[1][1]
    terms = [
        (i, j, k)
        for i in range(rows)
        for j in range(columns)
        for k in range(width)
        if k <= i
    ]
    expected_reads = len({(i, k) for i, _, k in terms}) + len(
        {(k, j) for _, j, k in terms}
    )
    assert (
        ArtifactDocument(result.output_dir, result.analysis).require_mapping(
            "total"
        )["macs"]
        == rows * width * columns
    )
    assert report.dense_macs == rows * width * columns
    assert report.macs == len(terms) == 30
    assert sum(report.external_read_elements.values()) == expected_reads
    assert report.external_read_bytes == 4 * expected_reads
    assert report.output_write_bytes == rows * columns * 4
    assert report.resource_work["mfma"]["fp32->fp32"] == 60
    mask = next(
        layer for layer in report.layers.values() if layer.target == "tril"
    )
    assert mask.output_writes == (rows * width,)
    assert mask.zero_initializations == rows * width - len(
        {(i, k) for i, _, k in terms}
    )
    assert (
        report.views[AnalysisView.CACHE_AWARE].status
        is ViewAvailability.UNAVAILABLE
    )
    assert (
        report.views[AnalysisView.IDEAL_FUSED].status
        is ViewAvailability.AVAILABLE
    )
    assert not result.record.sol_score_eligible
    assert not result.record.publication_eligible
    restored = load_projection(
        result.output_dir,
        record_sha256=sha256_file(result.output_dir / REUSE_FILENAME),
        projection_sha256=result.record.projection_sha256,
    )
    assert restored == result
    assert (result.output_dir / DEMAND_FILENAME).is_file()


@pytest.mark.parametrize(
    "constraint,reason",
    [
        (DemandRequest(), DemandReason.NUMERICS),
        (
            DemandRequest(
                numerical_domain=NumericalDomain.FINITE,
                budget=DemandBudget(cells=1),
            ),
            DemandReason.CELLS,
        ),
        (
            DemandRequest(
                numerical_domain=NumericalDomain.FINITE,
                budget=DemandBudget(terms=1),
            ),
            DemandReason.TERMS,
        ),
    ],
    ids=["ieee", "coordinate-budget", "term-budget"],
)
def test_declined_scenarios_retain_dense_without_fake_restricted_zero(
    triangle_source, tmp_path, constraint, reason
):
    _, identity = triangle_source
    result = ReuseSession(identity).project(
        ProjectionRequest(
            architecture=PROFILE,
            output_dir=tmp_path / "declined",
            restricted=constraint,
        )
    )
    assert result.restricted is not None
    assert result.restricted.reason is reason
    assert result.restricted.macs is None
    assert not result.restricted.views
    assert ArtifactDocument(result.output_dir, result.analysis).require_mapping(
        "total"
    )["macs"] in (48, 70)


def _intrinsic(
    reference, path, output, shapes, sample_id="F06-D", factory=None
):
    base = get_sample(sample_id).request(path, sha256_file(Path(__file__)))

    def inputs(seed):
        generator = torch.Generator().manual_seed(seed)
        return tuple(
            torch.randn(shape, generator=generator) for shape in shapes
        )

    conversion = replace(
        base, reference=reference, input_factory=factory or inputs
    )
    ready = audit_conversion(
        ConversionReadinessRequest(
            conversion=conversion,
            architecture=PROFILE,
            output_dir=output,
            require_verified_audit=False,
        )
    )
    assert ready.ready, ready
    analyzer = IRGraphAnalyzer()
    intrinsic = analyzer.prepare_intrinsic(
        AnalysisJob(
            graph_path=output / path.graph_filename,
            output_dir=output / "counts",
            precision="fp32",
            copy_graph=False,
            strict=True,
            architecture=PROFILE,
            orojenesis_runner=None,
            require_orojenesis=False,
        )
    )
    assert intrinsic is not None
    dense = analyzer.project_intrinsic(
        intrinsic,
        output_dir=output / "dense",
        architecture=ArchitectureProfile.load(PROFILE),
        write_output=False,
    )
    return intrinsic, dense


@pytest.mark.parametrize("path", list(IRPath))
@pytest.mark.parametrize(
    "shared",
    [False, True],
    ids=["partial-consumer", "shared-materialized-output"],
)
def test_backward_selection_and_shared_output_union(path, shared, tmp_path):
    def reference(a, b):
        value = a @ b
        return (value[:2], value) if shared else value[:2]

    intrinsic, dense = _intrinsic(
        reference, path, tmp_path / "source", ((4, 5), (5, 3))
    )
    report = demand_report(
        intrinsic,
        FINITE,
        dense,
        restrict_computation(intrinsic, FINITE),
        ArchitectureProfile.load(PROFILE),
    )
    assert dense["total"]["macs"] == 60
    assert report.macs == (60 if shared else 30)
    assert report.output_write_bytes == (48 if shared else 24)
    assert sum(report.external_read_elements.values()) == (35 if shared else 25)


@pytest.mark.parametrize("path", list(IRPath))
def test_pointwise_broadcast_demand_unions_both_static_slices(path, tmp_path):
    def reference(a, b):
        value = a + b
        return value[:2], value[1:3]

    intrinsic, dense = _intrinsic(
        reference, path, tmp_path / "source", ((4, 5), (5,))
    )
    report = demand_report(
        intrinsic,
        FINITE,
        dense,
        restrict_computation(intrinsic, FINITE),
        ArchitectureProfile.load(PROFILE),
    )
    addition = next(
        layer for layer in report.layers.values() if layer.target == "add"
    )
    assert addition.coordinate_work == 15
    assert addition.input_reads == (15, 5)
    assert addition.resource_work["valu"]["fp32"] == 15
    assert sum(report.external_read_elements.values()) == 20
    assert report.output_write_bytes == 60


@pytest.mark.parametrize("path", list(IRPath))
def test_log_softmax_breaks_zero_inference_and_copy_retains_full_cost(
    path, tmp_path
):
    def reference(a, b):
        copied = a.clone()
        return torch.log_softmax(torch.tril(copied), dim=-1) @ b

    intrinsic, dense = _intrinsic(
        reference, path, tmp_path / "source", ((4, 4), (4, 3))
    )
    report = demand_report(
        intrinsic,
        FINITE,
        dense,
        restrict_computation(intrinsic, FINITE),
        ArchitectureProfile.load(PROFILE),
    )
    assert report.macs == 48
    copy = next(
        layer for layer in report.layers.values() if layer.target == "clone"
    )
    assert copy.boundary is DemandReason.COPY
    assert copy.input_reads == copy.output_writes == (16,)
    normalization = next(
        layer
        for layer in report.layers.values()
        if layer.target == "log_softmax"
    )
    assert normalization.boundary in {
        DemandReason.EFFECT,
        DemandReason.UNSUPPORTED,
    }
    assert report.external_read_bytes is None


@pytest.mark.parametrize(
    "value", [float("inf"), float("nan")], ids=["infinity", "nan"]
)
def test_ieee_zero_elimination_has_explicit_counterexamples(value):
    mask = torch.tril(torch.ones(2, 2))
    weight = torch.tensor([[1.0], [value]])
    assert torch.isnan((mask @ weight)[0, 0])
    assert (mask[0, :1] @ weight[:1])[0] == 1.0


def test_invalid_budget_is_not_an_unsupported_result():
    with pytest.raises(ValueError, match="greater than"):
        DemandBudget(cells=0)
    with pytest.raises(ValueError, match="finite"):
        DemandBudget(seconds=float("inf"))


def test_time_checked_inside_enumeration_and_empty_huge_axis_is_cheap(
    monkeypatch,
):
    work = DemandWork(DemandBudget(), deadline=10.0)
    assert list(coordinates((0, 10**100), work)) == []
    calls = iter((0.0, 100.0))
    monkeypatch.setattr(
        "solar.analysis.demand_plan.monotonic", lambda: next(calls)
    )
    consumed = []
    with pytest.raises(DemandDeclinedError) as caught:
        consumed.extend(
            checked_items(range(2 * DEMAND_TIME_CHECK_INTERVAL), work)
        )
    assert caught.value.reason is DemandReason.TIME
    assert len(consumed) == DEMAND_TIME_CHECK_INTERVAL


def test_restricted_scan_inverse_and_resume_keep_constraints(
    triangle_source, tmp_path, monkeypatch
):
    _, identity = triangle_source
    request = ScanRequest(
        job=ReuseJob(
            source=identity,
            projection=ProjectionRequest(
                architecture=PROFILE,
                output_dir=tmp_path / "scan",
                restricted=FINITE,
            ),
        ),
        view=AnalysisView.IDEAL_FUSED,
        computation=ComputationScenario.RESTRICTED,
        axes=(ScanAxis(kind=AxisKind.BANDWIDTH_SCALE, values=(1, 2)),),
    )
    result = scan_profiles(request)
    assert all(point.status is PointStatus.ANALYZED for point in result.points)
    assert result.identity_inputs["computation"] == "restricted"
    point = result.points[0]
    assert point.result is not None and point.estimate is not None
    assert point.estimate.seconds is not None
    inverse = query_resources(
        InverseRequest(
            source=point.result,
            view=AnalysisView.IDEAL_FUSED,
            computation=ComputationScenario.RESTRICTED,
            target_seconds=point.estimate.seconds,
        )
    )
    assert inverse.status is InverseStatus.AVAILABLE
    assert inverse.computation is ComputationScenario.RESTRICTED
    assert sum(inverse.constraints["mfma"].work_by_mode.values()) == 60
    reference = ScanReference(
        directory=request.job.projection.output_dir,
        record_sha256=sha256_file(
            request.job.projection.output_dir / "scan.json"
        ),
    )
    resumed_request = request.model_copy(
        update={
            "job": request.job.model_copy(
                update={
                    "projection": request.job.projection.model_copy(
                        update={"output_dir": tmp_path / "resume"}
                    )
                }
            ),
            "resume": reference,
        }
    )

    def forbidden(*args, **kwargs):
        pytest.fail("complete resumed points cannot reproject")

    monkeypatch.setattr(ReuseSession, "project", forbidden)
    resumed = scan_profiles(resumed_request)
    assert all(point.resumed for point in resumed.points)
    assert [point.result for point in resumed.points] == [
        point.result for point in result.points
    ]
    changed = resumed_request.model_copy(
        update={
            "job": resumed_request.job.model_copy(
                update={
                    "projection": resumed_request.job.projection.model_copy(
                        update={
                            "output_dir": tmp_path / "changed",
                            "restricted": DemandRequest(),
                        }
                    )
                }
            )
        }
    )
    with pytest.raises(ValueError, match="identity differs"):
        scan_profiles(changed)


def test_unavailable_restricted_inverse_never_uses_dense(
    triangle_source, tmp_path
):
    _, identity = triangle_source
    result = ReuseSession(identity).project(
        ProjectionRequest(
            architecture=PROFILE,
            output_dir=tmp_path / "ieee",
            restricted=DemandRequest(),
        )
    )
    source = ProjectionReference(
        directory=result.output_dir,
        record_sha256=sha256_file(result.output_dir / REUSE_FILENAME),
        projection_sha256=result.record.projection_sha256,
    )
    inverse = query_resources(
        InverseRequest(
            source=source,
            view=AnalysisView.IDEAL_FUSED,
            computation=ComputationScenario.RESTRICTED,
            target_seconds=1.0,
        )
    )
    assert inverse.status is InverseStatus.UNAVAILABLE
    assert not inverse.constraints
    dense = query_resources(
        InverseRequest(
            source=source, view=AnalysisView.IDEAL_FUSED, target_seconds=1.0
        )
    )
    assert dense.status is InverseStatus.AVAILABLE


def test_restricted_artifact_corruption_and_omission_are_rejected(
    triangle_source, tmp_path
):
    _, identity = triangle_source
    result = ReuseSession(identity).project(
        ProjectionRequest(
            architecture=PROFILE,
            output_dir=tmp_path / "restrict",
            restricted=FINITE,
        )
    )
    path = result.output_dir / DEMAND_FILENAME
    saved = path.read_bytes()
    path.write_bytes(saved + b" ")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_projection(
            result.output_dir,
            record_sha256=sha256_file(result.output_dir / REUSE_FILENAME),
            projection_sha256=result.record.projection_sha256,
        )
    path.unlink()
    with pytest.raises(ValueError, match="inventory differs"):
        load_projection(
            result.output_dir,
            record_sha256=sha256_file(result.output_dir / REUSE_FILENAME),
            projection_sha256=result.record.projection_sha256,
        )


@pytest.mark.parametrize("path", list(IRPath))
def test_budget_before_coordinate_allocation_and_cheap_huge_dense(
    path, tmp_path, monkeypatch
):
    intrinsic, _ = _intrinsic(
        lambda a, b: torch.tril(a) @ b,
        path,
        tmp_path / "triangle",
        ((4, 4), (4, 3)),
    )

    def forbidden(*args, **kwargs):
        pytest.fail(
            "coordinate iteration must not start after budget rejection"
        )

    monkeypatch.setattr("solar.analysis.demand_plan.coordinates", forbidden)
    with pytest.raises(DemandDeclinedError) as caught:
        restrict_computation(
            intrinsic,
            DemandRequest(
                numerical_domain=NumericalDomain.FINITE,
                budget=DemandBudget(cells=1),
            ),
        )
    assert caught.value.reason is DemandReason.CELLS
    huge = 10**100
    # No transformation site: only constant-size count metadata is inspected.
    matrix = next(layer for layer in intrinsic.layers if layer.compute.macs)
    changed = replace(matrix, compute=replace(matrix.compute, macs=huge))
    assert (
        restrict_computation(replace(intrinsic, layers=(changed,)), FINITE)
        is None
    )


@pytest.mark.parametrize("path", list(IRPath))
def test_restricted_numeric_and_gradient_oracle_preserves_nonzero_terms(
    path, tmp_path
):
    sample = get_sample("F06-D")
    intrinsic, dense = _intrinsic(
        sample.reference(), path, tmp_path / "source", sample.shapes
    )
    report = demand_report(
        intrinsic,
        FINITE,
        dense,
        restrict_computation(intrinsic, FINITE),
        ArchitectureProfile.load(PROFILE),
    )
    assert report.macs == 30
    left, right = (tensor.requires_grad_() for tensor in sample.inputs(29))
    expected = sample.reference()(left, right)
    assert isinstance(expected, torch.Tensor)
    rows = [
        torch.stack(
            [
                sum(
                    (
                        left[i, k] * right[k, j]
                        for k in range(left.shape[1])
                        if k <= i
                    ),
                    start=left.new_zeros(()),
                )
                for j in range(right.shape[1])
            ]
        )
        for i in range(left.shape[0])
    ]
    actual = torch.stack(rows)
    torch.testing.assert_close(actual, expected)
    wanted = torch.autograd.grad(
        expected.sum(), (left, right), retain_graph=True
    )
    observed = torch.autograd.grad(actual.sum(), (left, right))
    for x, y in zip(observed, wanted, strict=True):
        torch.testing.assert_close(x, y)


@pytest.mark.parametrize("path", list(IRPath))
@pytest.mark.parametrize(
    "upper,diagonal",
    [(False, -20), (False, 20), (True, -1), (False, 2)],
    ids=["empty", "full", "upper-negative", "lower-positive"],
)
def test_rectangular_mask_domain_and_zero_initialization(
    path, upper, diagonal, tmp_path
):
    def reference(a, b):
        masked = (
            torch.triu(a, diagonal=diagonal)
            if upper
            else torch.tril(a, diagonal=diagonal)
        )
        return masked @ b

    intrinsic, dense = _intrinsic(
        reference, path, tmp_path / "source", ((3, 5), (5, 2))
    )
    report = demand_report(
        intrinsic,
        FINITE,
        dense,
        restrict_computation(intrinsic, FINITE),
        ArchitectureProfile.load(PROFILE),
    )
    pairs = {
        (i, k)
        for i in range(3)
        for k in range(5)
        if (k - i >= diagonal if upper else k - i <= diagonal)
    }
    assert report.macs == len(pairs) * 2
    assert report.dense_macs == 30
    mask = next(
        layer
        for layer in report.layers.values()
        if layer.target in {"tril", "triu"}
    )
    assert mask.output_writes == (15,)
    assert mask.zero_initializations == 15 - len(pairs)
    contraction = next(
        layer for layer in report.layers.values() if layer.dense_macs
    )
    assert contraction.output_writes == (6,)
    assert contraction.zero_initializations == 2 * len(
        {i for i in range(3) if not any(row == i for row, _ in pairs)}
    )


@pytest.mark.parametrize("path", list(IRPath))
def test_time_budget_reports_declined_without_estimate(
    path, tmp_path, monkeypatch
):
    intrinsic, _ = _intrinsic(
        lambda a, b: torch.tril(a) @ b,
        path,
        tmp_path / "source",
        ((4, 4), (4, 3)),
    )
    monkeypatch.setattr("solar.analysis.demand.monotonic", lambda: 0.0)
    monkeypatch.setattr("solar.analysis.demand_plan.monotonic", lambda: 100.0)
    with pytest.raises(DemandDeclinedError) as caught:
        restrict_computation(intrinsic, FINITE)
    assert caught.value.reason is DemandReason.TIME


@pytest.mark.parametrize("path", list(IRPath))
def test_returned_external_alias_does_not_claim_materialization_traffic(
    path, tmp_path
):
    intrinsic, dense = _intrinsic(
        lambda value: value[:2], path, tmp_path / "source", ((4, 5),)
    )
    report = demand_report(
        intrinsic,
        FINITE,
        dense,
        restrict_computation(intrinsic, FINITE),
        ArchitectureProfile.load(PROFILE),
    )
    assert report.read_reasons == (DemandReason.OUTPUT_ALIAS,)
    assert (
        report.external_read_bytes is None and report.output_write_bytes is None
    )
    assert (
        report.views[AnalysisView.IDEAL_FUSED].status
        is ViewAvailability.UNAVAILABLE
    )


@pytest.mark.parametrize("path", list(IRPath))
@pytest.mark.parametrize(
    "sample_id,reason",
    [
        ("F07-D", DemandReason.INDEX),
        ("F08-D", DemandReason.EFFECT),
        ("F05-D", DemandReason.DTYPE),
    ],
    ids=["dynamic-index", "observable-mutation", "nonfloating-payload"],
)
def test_unsupported_boundaries_keep_costs_and_prevent_read_claim(
    path, sample_id, reason, tmp_path
):
    sample = get_sample(sample_id)

    def indexed(value, indices):
        return torch.tril(value).index_select(0, indices)

    def mixed(value, mask):
        return torch.tril(value.float()) + mask.float()

    reference = (
        indexed
        if sample_id == "F07-D"
        else mixed
        if sample_id == "F05-D"
        else sample.reference()
    )
    intrinsic, dense = _intrinsic(
        reference,
        path,
        tmp_path / "source",
        sample.shapes,
        sample_id=sample_id,
        factory=sample.inputs,
    )
    report = demand_report(
        intrinsic,
        FINITE,
        dense,
        restrict_computation(intrinsic, FINITE),
        ArchitectureProfile.load(PROFILE),
    )
    assert any(layer.boundary is reason for layer in report.layers.values())
    assert report.external_read_bytes is None
    assert (
        report.views[AnalysisView.IDEAL_FUSED].status
        is ViewAvailability.UNAVAILABLE
    )
    for name, layer in report.layers.items():
        if layer.boundary is not None:
            assert (
                layer.resource_work
                == dense["layers"][name]["resources"]["work"]
            )
            assert layer.unfused_bytes == dense["layers"][name]["unfused_bytes"]
