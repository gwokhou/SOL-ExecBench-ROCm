"""Keep the original gate's decision out of the independent numeric label."""

import numpy as np

from scripts.internal.solar.e2_successor_labels import (
    NumericLabel,
    assess_numeric_call,
)
from sol_execbench.core.integrity import sha256_file


def test_a_passed_gate_can_still_have_an_independently_observed_violation(
    tmp_path,
):
    path = tmp_path / "actual.npz"
    np.savez_compressed(
        path,
        reference=np.array([1.0]),
        actual=np.array([2.0]),
        oracle_rounded=np.array([1.0]),
    )
    status = assess_numeric_call(
        path,
        {
            "artifact_sha256": sha256_file(path),
            "independent_oracle_points": 1,
            "incumbent_numeric_passed": True,
            "original_numeric_passed": False,
            "oracle_reference_original_numeric_passed": True,
            "witness_certified": False,
        },
        atol=0.125,
        rtol=0,
    )
    assert status.original_numeric_passed is True
    assert status.recomputed_numeric_passed is False
    assert status.label is NumericLabel.VIOLATION
    assert status.margin is not None
    assert status.margin.upper < 0


def test_reference_disagreement_keeps_an_ambiguous_rejection_unresolved(
    tmp_path,
):
    path = tmp_path / "actual.npz"
    np.savez_compressed(
        path,
        reference=np.array([1.25]),
        actual=np.array([1.0]),
        oracle_rounded=np.array([1.0]),
    )
    status = assess_numeric_call(
        path,
        {
            "artifact_sha256": sha256_file(path),
            "independent_oracle_points": 1,
            "incumbent_numeric_passed": False,
            "original_numeric_passed": False,
            "oracle_reference_original_numeric_passed": False,
        },
        atol=0.125,
        rtol=0,
    )
    assert status.original_numeric_passed is False
    assert status.reference_disagreement is True
    assert status.label is NumericLabel.INDETERMINATE
