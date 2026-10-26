"""Actual candidate extraction and unchanged original-oracle verification on CPU."""

import json
from dataclasses import replace
from functools import partial
from pathlib import Path

import pytest
import torch
import yaml

from scripts.internal.solar.base01_records import PROFILE
from scripts.internal.solar.base01_samples import get_sample
from solar.analysis.demand_contracts import (
    ComputationScenario,
    DemandRequest,
    NumericalDomain,
)
from solar.analysis.order_contracts import (
    ORDER_FILENAME,
    CandidateStatus,
    OrderBudget,
    OrderReference,
)
from solar.analysis.order_domain import OrderDomainError, matrix_chain
from solar.analysis.views import AnalysisView
from solar.api import (
    AnalysisRequest,
    AnalysisResult,
    ConversionReadinessRequest,
    analyze,
    audit_conversion,
)
from solar.artifacts import sha256_file, stable_json_checksum
from solar.artifacts.document import load_yaml_artifact
from solar.artifacts.names import (
    CONVERSION_ATTESTATION_FILENAME,
    MANIFEST_FILENAME,
)
from solar.ir.contracts import IRPath
from solar.order import search_contractions
from solar.order_contracts import OrderSearchRequest
from solar.pipeline.order_artifacts import load_order_result
from solar.pipeline.verified_program import load_verified_program
from solar.reuse_contracts import (
    ProjectionRequest,
    ReuseJob,
    VerifiedProgramRequest,
)
from solar.rocm.architecture import ArchitectureProfile


def source_request(conversion, output):
    result = analyze(
        AnalysisRequest(
            conversion=conversion,
            architecture=PROFILE,
            output_dir=output,
            precision="fp32",
            require_verified_audit=False,
        )
    )
    assert isinstance(result, AnalysisResult), result
    receipt = yaml.safe_load(
        (output / CONVERSION_ATTESTATION_FILENAME).read_text()
    )
    return VerifiedProgramRequest(
        directory=output,
        manifest_sha256=sha256_file(output / MANIFEST_FILENAME),
        reference_sha256=conversion.reference_sha256,
        execution_sha256=stable_json_checksum(
            receipt["predicate"]["execution"]
        ),
        ir_path=conversion.ir_path,
        verification=conversion.verification,
    )


@pytest.fixture(
    scope="module",
    params=[(sample, path) for sample in ("F03-D", "F03-Q") for path in IRPath],
    ids=lambda p: f"{p[0]}-{p[1].value}",
)
def chain_source(request, tmp_path_factory):
    sample_id, path = request.param
    conversion = get_sample(sample_id).request(
        path, sha256_file(Path(__file__))
    )
    source = source_request(
        conversion, tmp_path_factory.mktemp("order-source") / "full"
    )
    return sample_id, conversion, source


def order_request(source, output, **kwargs):
    profile = ArchitectureProfile.load(PROFILE).to_dict()
    profile["resource_limits"]["mfma"]["fp32->fp32"] = 100.0
    return OrderSearchRequest(
        job=ReuseJob(
            source=source,
            projection=ProjectionRequest(
                architecture=profile, output_dir=output
            ),
        ),
        view=AnalysisView.IDEAL_FUSED,
        **kwargs,
    )


def test_actual_f03_candidates_reprice_verify_and_resume(
    chain_source, tmp_path
):
    sample_id, conversion, source = chain_source
    request = order_request(source, tmp_path / "search")
    result = search_contractions(conversion, request)
    assert result.total_candidates == 2 and result.unexplored_candidates == 0
    assert result.exhaustive_verified_domain
    expected = [84, 70] if sample_id == "F03-D" else [165, 224]
    assert [item.cost.macs for item in result.candidates] == expected
    assert all(
        item.status is CandidateStatus.ANALYZED for item in result.candidates
    )
    assert (
        result.selected_candidate_id
        == result.candidates[expected.index(min(expected))].candidate_id
    )
    assert not result.performance_authority and not result.publication_eligible
    for item in result.candidates:
        assert item.views is not None and item.bundle is not None
        assert item.views[request.view].seconds == pytest.approx(
            2 * item.cost.macs / 100
        )
        receipt = yaml.safe_load(
            (
                item.bundle.directory / "full" / CONVERSION_ATTESTATION_FILENAME
            ).read_text()
        )
        assert receipt["predicate"]["status"] == "passed"
        assert receipt["predicate"]["tolerance"]["verify_gradients"]
    pin = OrderReference(
        directory=request.job.projection.output_dir,
        record_sha256=sha256_file(
            request.job.projection.output_dir / ORDER_FILENAME
        ),
    )
    assert load_order_result(pin) == result
    resumed = request.model_copy(
        update={
            "resume": pin,
            "job": request.job.model_copy(
                update={
                    "projection": request.job.projection.model_copy(
                        update={"output_dir": tmp_path / "resumed"}
                    )
                }
            ),
        }
    )
    recovered = search_contractions(conversion, resumed)
    assert all(item.resumed for item in recovered.candidates)
    assert [item.bundle for item in recovered.candidates] == [
        item.bundle for item in result.candidates
    ]


def test_candidate_budget_retains_full_denominator(chain_source, tmp_path):
    _, conversion, source = chain_source
    result = search_contractions(
        conversion,
        order_request(
            source, tmp_path / "limited", budget=OrderBudget(max_candidates=1)
        ),
    )
    assert result.total_candidates == 2 and result.unexplored_candidates == 1
    assert not result.exhaustive_verified_domain
    assert result.candidates[0].original


def test_actual_work_rejected_before_input_factory(chain_source, tmp_path):
    _, conversion, source = chain_source

    def forbidden(seed):
        pytest.fail(
            "excessive candidate must be rejected before allocating inputs"
        )

    conversion = replace(conversion, input_factory=forbidden)
    result = search_contractions(
        conversion,
        order_request(
            source, tmp_path / "work", budget=OrderBudget(max_macs=1)
        ),
    )
    assert all(
        item.status is CandidateStatus.WORK_BUDGET and not item.attempted
        for item in result.candidates
    )
    assert result.unexplored_candidates == 2
    assert result.selected_candidate_id is None


def test_changed_oracle_policy_is_not_accepted(chain_source, tmp_path):
    _, conversion, source = chain_source
    altered = replace(
        conversion, verification=replace(conversion.verification, atol=0.5)
    )
    with pytest.raises(ValueError, match="policy differs"):
        search_contractions(
            altered, order_request(source, tmp_path / "invalid")
        )


def test_non_matrix_source_is_explicitly_unsupported(tmp_path):
    conversion = get_sample("F02-D").request(
        IRPath.MAKE_FX_ATEN, sha256_file(Path(__file__))
    )
    source = source_request(conversion, tmp_path / "source")
    with pytest.raises(OrderDomainError):
        matrix_chain(load_verified_program(source).graph.data)


def strict_inputs(seed):
    del seed
    return tuple(
        torch.tensor([[value]], dtype=torch.float32)
        for value in (1e20, 1e-20, 1e-20)
    )


@pytest.mark.parametrize("path", list(IRPath), ids=lambda path: path.value)
def test_reassociation_failure_retains_original_numerical_policy(
    path, tmp_path
):
    conversion = get_sample("F03-D").request(path, sha256_file(Path(__file__)))
    conversion = replace(
        conversion,
        input_factory=strict_inputs,
        verification=replace(
            conversion.verification,
            atol=0,
            rtol=0,
            max_error_cap=0,
            gradient_atol=0,
            gradient_rtol=0,
        ),
    )
    source = source_request(conversion, tmp_path / "strict-source")
    result = search_contractions(
        conversion, order_request(source, tmp_path / "strict-search")
    )
    original, changed = result.candidates
    assert original.status is CandidateStatus.ANALYZED
    assert changed.status is CandidateStatus.VERIFICATION_FAILED
    assert changed.views is None and changed.projection is None
    assert changed.bundle is not None and changed.message
    assert result.selected_candidate_id == original.candidate_id
    assert not (changed.bundle.directory / "full" / MANIFEST_FILENAME).exists()


@pytest.mark.parametrize(
    "finite", [True, False], ids=["explicit-finite", "default-ieee"]
)
def test_restricted_search_requires_its_own_available_scenario(
    chain_source, tmp_path, finite
):
    _, conversion, source = chain_source
    request = order_request(source, tmp_path / "restricted")
    restricted = DemandRequest(
        numerical_domain=NumericalDomain.FINITE
        if finite
        else NumericalDomain.IEEE
    )
    request = request.model_copy(
        update={
            "computation": ComputationScenario.RESTRICTED,
            "job": request.job.model_copy(
                update={
                    "projection": request.job.projection.model_copy(
                        update={"restricted": restricted}
                    )
                }
            ),
        }
    )
    result = search_contractions(conversion, request)
    assert all(
        item.status
        is (CandidateStatus.ANALYZED if finite else CandidateStatus.UNAVAILABLE)
        for item in result.candidates
    )
    assert (result.selected_candidate_id is not None) == finite
    assert result.exhaustive_verified_domain == finite


def test_missing_candidate_evidence_invalidates_recovery(
    chain_source, tmp_path
):
    _, conversion, source = chain_source
    request = order_request(
        source, tmp_path / "missing", budget=OrderBudget(max_candidates=1)
    )
    result = search_contractions(conversion, request)
    bundle = result.candidates[0].bundle
    assert bundle is not None
    (bundle.directory / "full" / CONVERSION_ATTESTATION_FILENAME).unlink()
    pin = OrderReference(
        directory=request.job.projection.output_dir,
        record_sha256=sha256_file(
            request.job.projection.output_dir / ORDER_FILENAME
        ),
    )
    with pytest.raises(ValueError, match="inventory"):
        load_order_result(pin)


def test_wrong_global_selection_claim_is_rejected(chain_source, tmp_path):
    _, conversion, source = chain_source
    request = order_request(
        source, tmp_path / "limited-claim", budget=OrderBudget(max_candidates=1)
    )
    result = search_contractions(conversion, request)
    path = request.job.projection.output_dir / ORDER_FILENAME
    payload = result.model_dump(mode="json")
    payload["exhaustive_verified_domain"] = True
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="selection or coverage"):
        load_order_result(
            OrderReference(
                directory=path.parent, record_sha256=sha256_file(path)
            )
        )


def test_profile_or_budget_drift_cannot_recover_old_candidates(
    chain_source, tmp_path
):
    _, conversion, source = chain_source
    request = order_request(
        source, tmp_path / "first", budget=OrderBudget(max_candidates=1)
    )
    search_contractions(conversion, request)
    pin = OrderReference(
        directory=request.job.projection.output_dir,
        record_sha256=sha256_file(
            request.job.projection.output_dir / ORDER_FILENAME
        ),
    )
    changed = request.model_copy(
        update={
            "resume": pin,
            "budget": OrderBudget(max_candidates=2),
            "job": request.job.model_copy(
                update={
                    "projection": request.job.projection.model_copy(
                        update={"output_dir": tmp_path / "wrong-identity"}
                    )
                }
            ),
        }
    )
    with pytest.raises(ValueError, match="search identity differs"):
        search_contractions(conversion, changed)


def four_matrices(a, b, c, d):
    return ((a @ b) @ c) @ d


def four_inputs(seed):
    generator = torch.Generator().manual_seed(seed)
    return tuple(
        torch.randn(shape, generator=generator)
        for shape in ((2, 3), (3, 4), (4, 5), (5, 2))
    )


@pytest.mark.parametrize("path", list(IRPath), ids=lambda path: path.value)
def test_four_matrix_search_matches_five_independent_costs(path, tmp_path):
    conversion = get_sample("F03-D").request(path, sha256_file(Path(__file__)))
    conversion = replace(
        conversion, reference=four_matrices, input_factory=four_inputs
    )
    source = source_request(conversion, tmp_path / "four-source")
    result = search_contractions(
        conversion, order_request(source, tmp_path / "four-search")
    )
    assert result.total_candidates == 5 and result.exhaustive_verified_domain
    assert sorted(item.cost.macs for item in result.candidates) == [
        76,
        80,
        84,
        102,
        110,
    ]
    winner = next(
        item
        for item in result.candidates
        if item.candidate_id == result.selected_candidate_id
    )
    assert winner.cost.macs == 76


def test_unexpected_candidate_invariant_is_retained_and_raised(
    chain_source, tmp_path, monkeypatch
):
    from solar import order

    _, conversion, source = chain_source
    request = order_request(source, tmp_path / "invariant")

    def broken(*args):
        raise RuntimeError("unexpected accounting invariant")

    monkeypatch.setattr(order, "run_candidate", broken)
    with pytest.raises(RuntimeError, match="unexpected accounting invariant"):
        search_contractions(conversion, request)
    raw = json.loads(
        (request.job.projection.output_dir / ORDER_FILENAME).read_text()
    )
    assert raw["candidates"][0]["reason"] == "invariant_failure"
    assert raw["candidates"][1]["status"] == "not_run"
    assert not raw["exhaustive_verified_domain"]


def test_equivalent_policy_sequence_containers_are_normalized(
    chain_source, tmp_path
):
    _, conversion, source = chain_source
    conversion = replace(
        conversion,
        verification=replace(
            conversion.verification,
            seeds=list(conversion.verification.seeds),
            patterns=list(conversion.verification.patterns),
        ),
    )
    result = search_contractions(
        conversion,
        order_request(
            source, tmp_path / "sequence-policy", budget=OrderBudget(max_macs=1)
        ),
    )
    assert all(
        item.status is CandidateStatus.WORK_BUDGET for item in result.candidates
    )


def shaped_inputs(seed, shapes):
    generator = torch.Generator().manual_seed(seed)
    return tuple(torch.randn(shape, generator=generator) for shape in shapes)


def repeated_input(a):
    return a @ a


def transposed_input(a, b):
    return a.transpose(0, 1) @ b


@pytest.mark.parametrize("path", list(IRPath), ids=lambda path: path.value)
@pytest.mark.parametrize(
    "reference,shapes",
    [
        (get_sample("F01-D").reference(), ((2, 3, 5), (2, 5, 4))),
        (repeated_input, ((3, 3),)),
        (transposed_input, ((3, 5), (3, 4))),
    ],
    ids=["batch-outside-domain", "repeated-source", "transpose-outside-domain"],
)
def test_verified_ir_outside_order_domain_is_explicitly_rejected(
    path, reference, shapes, tmp_path
):
    conversion = get_sample("F01-D").request(path, sha256_file(Path(__file__)))
    conversion = replace(
        conversion,
        reference=reference,
        input_factory=partial(shaped_inputs, shapes=shapes),
    )
    output = tmp_path / "verified-ir"
    readiness = audit_conversion(
        ConversionReadinessRequest(
            conversion=conversion,
            architecture=PROFILE,
            output_dir=output,
            require_verified_audit=False,
        )
    )
    assert readiness.ready, readiness
    graph = load_yaml_artifact(output / path.graph_filename)
    with pytest.raises(OrderDomainError):
        matrix_chain(graph.data)


def test_total_budget_after_preparation_preserves_every_unrun_candidate(
    chain_source, tmp_path, monkeypatch
):
    from solar import order
    from solar.reuse import ReuseSession

    _, conversion, source = chain_source
    clock = {"seconds": 0.0}
    project = ReuseSession.project

    def prepared_then_expired(self, request):
        result = project(self, request)
        clock["seconds"] = 301.0
        return result

    monkeypatch.setattr(order, "monotonic", lambda: clock["seconds"])
    monkeypatch.setattr(ReuseSession, "project", prepared_then_expired)
    result = search_contractions(
        conversion, order_request(source, tmp_path / "expired")
    )
    assert result.stop_reason == "total_budget_exhausted"
    assert result.unexplored_candidates == result.total_candidates == 2
    assert all(
        item.status is CandidateStatus.NOT_RUN for item in result.candidates
    )
    assert result.baseline is not None and result.selected_candidate_id is None


def test_rules_drift_cannot_finish_an_unexecuted_search(
    chain_source, tmp_path, monkeypatch
):
    from solar import order
    from solar.reuse_contracts import ReuseRulesChangedError

    _, conversion, source = chain_source
    original = order.analysis_rules_identity()
    identities = iter((original, dict.fromkeys(original, "f" * 64)))
    monkeypatch.setattr(
        order, "analysis_rules_identity", lambda: next(identities)
    )
    request = order_request(
        source, tmp_path / "rules-drift", budget=OrderBudget(max_macs=1)
    )
    with pytest.raises(ReuseRulesChangedError, match="rules changed"):
        search_contractions(conversion, request)
    raw = json.loads(
        (request.job.projection.output_dir / ORDER_FILENAME).read_text()
    )
    assert raw["stop_reason"] == "rules_changed_during_search"
    assert not raw["exhaustive_verified_domain"]
