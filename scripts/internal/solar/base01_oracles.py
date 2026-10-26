"""Independent finite-set and rational arithmetic oracles, not SOLAR features.

This module intentionally imports no production graph, geometry, accounting,
verification, cache or inverse-query implementation. These fixtures do not
probe whether a production capability exists or has passed acceptance.
"""

from __future__ import annotations

import math
from enum import StrEnum
from fractions import Fraction
from typing import Any


class ImplementationStatus(StrEnum):
    """Keep oracle results separate from unexecuted production acceptance."""

    NOT_EVALUATED = "not_evaluated"


def unique_reads(*regions: set[tuple[str, int, int]]) -> int:
    """Enumerate (base, effect epoch, offset) tokens in tiny oracle domains."""
    return len(set().union(*regions))


def read_tokens(
    start: int, stop: int, *, base: str = "x", epoch: int = 0
) -> set[tuple[str, int, int]]:
    """Build concrete tokens without a production region resolver."""
    return {(base, epoch, index) for index in range(start, stop)}


def geometry_oracles() -> list[dict[str, object]]:
    """Materialize all ten frozen geometry cases, including rejection inputs."""
    rows = read_tokens(0, 16)
    columns = {("x", 0, row * 8 + col) for row in range(8) for col in range(2)}
    invalid_interval, base_domain = range(129), range(128)
    declared_count, actual_count = 64, len(range(32))
    base_shape, conflicting_shape = (128,), (64,)
    regions, budget = ((0, 8), (16, 24), (32, 40)), 2
    invalid = [
        len(invalid_interval) > len(base_domain),
        actual_count != declared_count,
        base_shape != conflicting_shape,
        len(regions) > budget,
    ]
    specifications = [
        ("G01", unique_reads(read_tokens(0, 64), read_tokens(0, 64)), 64),
        ("G02", unique_reads(read_tokens(0, 64), read_tokens(32, 96)), 96),
        ("G03", unique_reads(read_tokens(0, 32), read_tokens(64, 96)), 64),
        ("G04", unique_reads(rows, columns), 28),
        ("G05", unique_reads(read_tokens(0, 16), read_tokens(32, 48)), 32),
        (
            "G06",
            unique_reads(read_tokens(0, 32), read_tokens(0, 32, base="y")),
            64,
        ),
        ("G07", [len(set(x)) for x in ((0, 7, 7, 31), (0, 7, 8, 31))], [3, 4]),
        (
            "G08",
            [
                len(range(128)[s])
                for s in (slice(-16, -4), slice(20, 20), slice(1, 12, 2))
            ],
            [12, 0, 6],
        ),
        ("G09", invalid, [True] * 4),
        (
            "G10",
            unique_reads(read_tokens(0, 8), read_tokens(0, 8, epoch=1)),
            16,
        ),
    ]
    return [
        {
            "id": name,
            "observed_oracle": observed,
            "expected_oracle": expected,
            "oracle_matches": observed == expected,
            "implementation_status": ImplementationStatus.NOT_EVALUATED,
        }
        for name, observed, expected in specifications
    ]


GEOMETRY_BOUNDARIES: dict[str, Any] = {
    "G06": "unproven_cross_input_alias_requires_conservative_or_rejected_result",
    "G07": "same_index_shape_has_different_sets_no_static_exact_claim",
    "G08": "unit_step_exact_nonunit_step_requires_explicit_support",
    "G09": {
        "bounds": [0, 129],
        "base_shape": [128],
        "declared_elements": 64,
        "actual_interval": [0, 32],
        "conflicting_shape": [64],
        "region_budget": 2,
        "regions": [[0, 8], [16, 24], [32, 40]],
        "expected": [
            "reject",
            "reject",
            "reject",
            "conservative_or_unavailable",
        ],
    },
    "G10": {
        "effect_epochs_must_not_merge": True,
        "atomic_requires_effect_boundary": True,
        "transpose_then_contiguous_requires_copy": True,
    },
}


def positive_finite(value: float) -> bool:
    """Validate an independent oracle's domain, not a future public API."""
    return math.isfinite(value) and value > 0


def model_oracles() -> list[dict[str, object]]:
    """Evaluate rational formulas without claiming production capability."""
    shared = Fraction(100, 10) + Fraction(60, 20)
    grid = [
        max(Fraction(120, 60 * compute), Fraction(80, 40 * bandwidth))
        for compute, bandwidth in ((1, 1), (1, 2), (2, 1), (2, 2))
    ]
    numerical = {
        "M01": ([max(8, 2) + max(2, 8), max(8 + 2, 2 + 8)], [16, 10]),
        "M02": (max(shared, 9, 12), Fraction(13)),
        "M06": (
            [Fraction(120, 2), Fraction(80, 2)],
            [Fraction(60), Fraction(40)],
        ),
        "M07": (shared / 10, Fraction(13, 10)),
        "M08": (
            [positive_finite(x) for x in (0, -1, math.nan, math.inf)],
            [False] * 4,
        ),
        "M09": (grid, [Fraction(2), Fraction(2), Fraction(2), Fraction(1)]),
    }
    records = []
    for index in range(1, 11):
        case_id = f"M{index:02}"
        record: dict[str, object] = {
            "id": case_id,
            "implementation_status": ImplementationStatus.NOT_EVALUATED,
            "contract": MODEL_CONTRACTS[case_id],
            "oracle_kind": "contract_only",
        }
        if case_id in numerical:
            observed, expected = numerical[case_id]
            record.update(
                oracle_kind="rational_or_domain",
                observed_oracle=str(observed),
                expected_oracle=str(expected),
                oracle_matches=observed == expected,
            )
        records.append(record)
    return records


MODEL_CONTRACTS: dict[str, Any] = {
    "M01": {
        "serial_operator_times": [[8, 2], [2, 8]],
        "expected_seconds": [16, 10],
    },
    "M02": {
        "shared_modes": [[100, 10], [60, 20]],
        "independent_seconds": 9,
        "dram_seconds": 12,
    },
    "M03": {
        "sample": "F03-D",
        "capacities_bytes": [64, 4096],
        "expected": "traffic_nonincreasing_or_unavailable",
        "real_mapper": "not_run",
    },
    "M04": {
        "samples": ["F01-D", "F01-Q", "F05-D", "F05-Q"],
        "equal_fields": ["work", "bytes", "seconds", "qualification"],
        "reuse_reference_calls": 0,
        "reuse_extractions": 0,
    },
    "M05": {
        "invalidate_variants": [
            "source",
            "ir",
            "shape",
            "dtype",
            "effects",
            "verification_policy",
            "execution_identity",
            "rules",
            "resource_model",
            "capacity",
            "mapper",
            "budget",
            "timeout",
            "tampered_artifact",
        ],
        "rate_only": "reuse_intrinsic_work_reprice_recheck_qualification",
    },
    "M06": {
        "work_ops": 120,
        "bytes": 80,
        "target_seconds": 2,
        "expected": "necessary_not_sufficient",
    },
    "M07": {
        "shared_modes": [[100, 10], [60, 20]],
        "target_seconds": 10,
        "common_multiplier": "13/10",
        "unfused": "unsupported_until_separate_solver",
    },
    "M08": {
        "invalid_targets": ["zero", "negative", "nan", "infinity"],
        "invalid_rates": ["zero", "negative"],
        "missing_mode": "reject",
    },
    "M09": {
        "compute_multipliers": [1, 2],
        "bandwidth_multipliers": [1, 2],
        "capacity_changes": False,
        "expected_seconds": [2, 2, 2, 1],
    },
    "M10": {
        "variants": [
            "capacity_change",
            "interruption",
            "partial_failure",
            "budget_exhaustion",
            "resume_identity_drift",
            "hypothetical_profile_with_old_audit",
        ],
        "keep_all_points": True,
        "publication_eligible": False,
    },
}


def f_expectations() -> dict[str, dict[str, int]]:
    """Freeze mathematical targets separately from observed analyzer totals."""
    return {
        "F01-D": {"macs": 60, "flops": 120, "fused_bytes": 188},
        "F01-Q": {"macs": 385, "flops": 770, "fused_bytes": 668},
        "F03-D": {"macs": 84},
        "F03-Q": {"macs": 165},
        "F04-D": {"fused_bytes": 392},
        "F04-Q": {"fused_bytes": 648},
        "F05-D": {"fused_bytes": 224},
        "F05-Q": {"fused_bytes": 147},
        "F06-D": {"macs": 48},
        "F06-Q": {"macs": 70},
    }


def resource_expectations() -> dict[str, dict[str, dict[str, int]]]:
    """Count scalar work even when a legacy FLOP total excludes non-MAC work."""
    return {
        "F02-D": {"valu": {"fp32": 15}, "reduction": {"fp32": 12}},
        "F02-Q": {"valu": {"fp32": 77}, "reduction": {"fp32": 70}},
        "F05-D": {
            "valu": {"fp32": 32},
            "conversion": {"bf16->fp32": 32, "bool->fp32": 32},
        },
        "F05-Q": {
            "valu": {"fp32": 21},
            "conversion": {"bf16->fp32": 21, "bool->fp32": 21},
        },
        "F08-D": {"valu": {"fp32": 8}},
        "F08-Q": {"valu": {"fp32": 12}},
    }
