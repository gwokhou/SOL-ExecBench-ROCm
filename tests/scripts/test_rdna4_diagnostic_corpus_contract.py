"""Contract checks for the governed RDNA4 diagnostic corpus design."""

from pathlib import Path
from types import SimpleNamespace

import pytest


def test_preregistration_rejects_packaged_candidate_contract_breach(
    load_script,
    tmp_path: Path,
) -> None:
    corpus = load_script(
        "scripts/internal/rdna4/build_rdna4_diagnostic_corpora.py",
    )

    with pytest.raises(
        ValueError,
        match=(
            r"point_fit-transformer_block-02 axis M=1056 violates "
            r"packaged candidate contract \[1, 1024\]"
        ),
    ):
        corpus._preregister(tmp_path, 220)

    assert not (tmp_path / "design.json").exists()


def test_axis_limits_match_packaged_candidate_sources(load_script) -> None:
    corpus = load_script(
        "scripts/internal/rdna4/build_rdna4_diagnostic_corpora.py",
    )
    expected = {
        corpus.WorkloadKind.SOFTMAX: (
            "kMaximumColumns",
            corpus.SOFTMAX_MAXIMUM_COLUMNS,
        ),
        corpus.WorkloadKind.CROSS_ENTROPY: (
            "kMaximumClasses",
            corpus.CROSS_ENTROPY_MAXIMUM_CLASSES,
        ),
        corpus.WorkloadKind.TRANSFORMER: (
            "kMaximumSequence",
            corpus.TRANSFORMER_MAXIMUM_SEQUENCE,
        ),
    }

    for family, (name, value) in expected.items():
        _, solution = corpus._definition_template(family)
        source = solution["sources"][0]["content"]
        assert f"constexpr int {name} = {value};" in source


def test_start_280_successor_is_valid_and_pair_disjoint(load_script) -> None:
    corpus = load_script(
        "scripts/internal/rdna4/build_rdna4_diagnostic_corpora.py",
    )
    successor = corpus._all_cases(corpus.PAIR_DISJOINT_SUCCESSOR_START)
    historical = [
        case for start in (100, 160, 220) for case in corpus._all_cases(start)
    ]

    corpus._validate_template_axis_contracts(successor)
    assert {case.workload_uuid for case in successor}.isdisjoint(
        case.workload_uuid for case in historical
    )
    transformer_sequences = {
        case.axes["M"]
        for case in successor
        if case.family is corpus.WorkloadKind.TRANSFORMER
    }
    assert len(transformer_sequences) == 60
    assert max(transformer_sequences) <= corpus.TRANSFORMER_MAXIMUM_SEQUENCE
    assert all(sequence % 2 == 1 for sequence in transformer_sequences)


def test_representative_successor_uses_audited_real_shape_neighborhoods(
    load_script,
) -> None:
    corpus = load_script(
        "scripts/internal/rdna4/build_rdna4_diagnostic_corpora.py",
    )
    successor = corpus._all_cases(corpus.REPRESENTATIVE_SUCCESSOR_START)
    earlier = [
        case
        for start in (100, 160, 220, 280)
        for case in corpus._all_cases(start)
    ]

    corpus._validate_design_contracts(successor)
    assert {case.workload_uuid for case in successor}.isdisjoint(
        case.workload_uuid for case in earlier
    )
    transformer = sorted(
        (
            case
            for case in successor
            if case.family is corpus.WorkloadKind.TRANSFORMER
        ),
        key=lambda case: case.global_index,
    )
    assert tuple(case.axes["M"] for case in transformer) == (
        corpus.TRANSFORMER_REPRESENTATIVE_SEQUENCE_LENGTHS
    )
    assert len(transformer) == 60
    assert all(case.axes["N"] == 768 for case in transformer)
    for anchor, neighborhood in corpus._TRANSFORMER_REALISM_NEIGHBORHOODS:
        assert len(neighborhood) == 4
        assert max(abs(sequence - anchor) for sequence in neighborhood) <= 7


def test_capacity_governed_successor_has_disjoint_real_shape_neighborhoods(
    load_script,
) -> None:
    corpus = load_script(
        "scripts/internal/rdna4/build_rdna4_diagnostic_corpora.py",
    )
    successor = corpus._all_cases(corpus.CAPACITY_GOVERNED_SUCCESSOR_START)
    earlier = [
        case
        for start in (100, 160, 220, 280, 340)
        for case in corpus._all_cases(start)
    ]

    corpus._validate_design_contracts(successor)
    assert {case.workload_uuid for case in successor}.isdisjoint(
        case.workload_uuid for case in earlier
    )
    transformer = sorted(
        (
            case
            for case in successor
            if case.family is corpus.WorkloadKind.TRANSFORMER
        ),
        key=lambda case: case.global_index,
    )
    assert tuple(case.axes["M"] for case in transformer) == (
        corpus.CAPACITY_GOVERNED_TRANSFORMER_SEQUENCE_LENGTHS
    )
    for (
        anchor,
        neighborhood,
    ) in corpus._CAPACITY_GOVERNED_TRANSFORMER_NEIGHBORHOODS:
        assert len(neighborhood) == 4
        assert max(abs(sequence - anchor) for sequence in neighborhood) <= 11


def test_future_transformer_generation_requires_an_authored_realism_policy(
    load_script,
) -> None:
    corpus = load_script(
        "scripts/internal/rdna4/build_rdna4_diagnostic_corpora.py",
    )

    with pytest.raises(
        ValueError,
        match="representative schedule is not authored",
    ):
        corpus._all_cases(460)


def test_capacity_policy_rejects_out_of_range_indexed_read(
    load_script,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus = load_script(
        "scripts/internal/rdna4/build_rdna4_diagnostic_corpora.py",
    )
    case = corpus.CaseSpec(
        phase="held_out",
        family=corpus.WorkloadKind.INDEXED_READ,
        index=0,
        global_index=460,
        axes={"M": 200_000, "N": 1024},
    )
    monkeypatch.setattr(corpus, "_all_cases", lambda _start: [case])
    policy = SimpleNamespace(
        applicability_min_bytes=64 * 2**20,
        applicability_max_bytes=512 * 2**20,
    )

    with pytest.raises(
        ValueError,
        match="held_out-indexed_read-00 working_set_bytes=819200000",
    ):
        corpus._validate_design_working_sets(460, policy)
