"""Classify retained E2 calls independently of the original gate's verdict."""

import math
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np

from scripts.internal.solar.e2_archive_arrays import read_output_arrays
from scripts.internal.solar.e2_reference_margin import (
    ReferenceMargin,
    reference_margin,
)
from sol_execbench.core.integrity import sha256_file


class NumericLabel(StrEnum):
    """Scope-limited verdict of the independent margin analysis."""

    WITHIN_TOLERANCE = "within_tolerance"
    VIOLATION = "violation"
    INDETERMINATE = "indeterminate"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True, kw_only=True)
class NumericAssessment:
    """Keep the certified label separate from the recorded evaluator result."""

    label: NumericLabel
    original_numeric_passed: bool | None
    stronger_guarantee_passed: bool | None
    reference_disagreement: bool | None
    margin: ReferenceMargin | None = None
    reason: str | None = None
    recomputed_numeric_passed: bool | None = None


def tolerances(checks: list[dict[str, Any]]) -> tuple[float, float]:
    """Use the original all-element E0 check from the verified session receipt."""
    (check,) = checks
    if (
        check["type"] != "numeric"
        or check["mode"] != "elementwise"
        or check["required_matched_ratio"] != 1
        or check["allow_negative_inf"] is not False
        or check["max_error_cap"] is not None
    ):
        raise ValueError(
            "E2 retained session does not use its original numeric contract"
        )
    atol, rtol = float(check["max_atol"]), float(check["max_rtol"])
    if any(not math.isfinite(v) or v < 0 for v in (atol, rtol)):
        raise ValueError("E2 retained tolerance is invalid")
    return atol, rtol


def assess_numeric_call(
    source: Path,
    row: dict[str, Any],
    *,
    atol: float,
    rtol: float,
) -> NumericAssessment:
    """Read actual arrays, verify integrity and assign an independent strict sign."""
    if sha256_file(source) != row["artifact_sha256"]:
        raise ValueError("E2 numeric assessment source hash changed")
    gate_field = (
        "incumbent_timing_validation_passed"
        if row.get("operation") == "timing_iteration"
        else "incumbent_numeric_passed"
    )
    incumbent = row.get(gate_field)
    recomputed = row.get("original_numeric_passed")
    stronger = row.get("stronger_guarantee_passed")
    reference = row.get("oracle_reference_original_numeric_passed")
    disagreement = None if reference is None else not reference
    if not row.get("independent_oracle_points"):
        return NumericAssessment(
            label=NumericLabel.UNAVAILABLE,
            original_numeric_passed=incumbent,
            stronger_guarantee_passed=stronger,
            reference_disagreement=disagreement,
            reason="oracle_certification_unavailable",
            recomputed_numeric_passed=recomputed,
        )
    with np.load(source, allow_pickle=False) as archive:
        arrays = read_output_arrays(archive, with_oracle=True)
        oracle = arrays.get("oracle_rounded")
        if oracle is None or not np.isfinite(oracle).all():
            return NumericAssessment(
                label=NumericLabel.UNAVAILABLE,
                original_numeric_passed=incumbent,
                stronger_guarantee_passed=stronger,
                reference_disagreement=disagreement,
                reason="finite_certified_oracle_unavailable",
                recomputed_numeric_passed=recomputed,
            )
        actual = arrays["actual"]
        observed_reference = arrays["reference"]
        if not np.isfinite(observed_reference).all():
            return NumericAssessment(
                label=NumericLabel.INDETERMINATE,
                original_numeric_passed=incumbent,
                stronger_guarantee_passed=stronger,
                reference_disagreement=disagreement,
                reason="nonfinite_reference",
                recomputed_numeric_passed=recomputed,
            )
        if not np.isfinite(actual).all():
            return NumericAssessment(
                label=NumericLabel.VIOLATION,
                original_numeric_passed=incumbent,
                stronger_guarantee_passed=stronger,
                reference_disagreement=disagreement,
                reason="nonfinite_candidate_under_original_contract",
                recomputed_numeric_passed=recomputed,
            )
        margin = reference_margin(
            actual, observed_reference, oracle, atol=atol, rtol=rtol
        )
    if margin.lower > 0:
        outcome = NumericLabel.WITHIN_TOLERANCE
    elif margin.upper < 0:
        outcome = NumericLabel.VIOLATION
    else:
        outcome = NumericLabel.INDETERMINATE
    return NumericAssessment(
        label=outcome,
        original_numeric_passed=incumbent,
        stronger_guarantee_passed=stronger,
        reference_disagreement=disagreement,
        margin=margin,
        recomputed_numeric_passed=recomputed,
    )
