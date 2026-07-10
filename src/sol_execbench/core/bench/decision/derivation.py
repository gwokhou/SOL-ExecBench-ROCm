# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Layer R decision hint derivation from static footprints and arch budgets.

Emits only the static-inferable Layer R subset (decision-modeling-research.md
§5): register spill, VGPR pressure, and LDS pressure. Wavefront/cache-line
alignment, workgroup-size, and barrier limits require block-size or access
pattern data the static footprint does not carry, so they stay deferred and are
never speculated (unknown -> not emitted). Dynamic register allocation (RDNA4+)
defeats static occupancy derivation, so it emits a single low-confidence note
with an explicit limitation (research §6.4).
"""

from __future__ import annotations

from collections.abc import Sequence

from sol_execbench.core.bench.decision.decision_models import (
    DecisionBottleneckClass,
    DecisionConfidence,
    DecisionHint,
    DecisionHintIdentity,
)
from sol_execbench.core.bench.static_kernel.evidence_models import (
    StaticResourceFootprint,
)
from sol_execbench.core.platform.arch_capabilities import ArchIsaBudget

# Heuristic ratios. Resource pressure is flagged when usage crosses 80% of the
# architected budget; occupancy corroboration uses the AMD SOL yellow threshold
# of 50% (decision-modeling-research.md §8.3).
_REGISTER_PRESSURE_RATIO = 0.8
_LDS_PRESSURE_RATIO = 0.8
_OCCUPANCY_LOW_RATIO = 0.5

_RECOMMENDATIONS = {
    DecisionBottleneckClass.SPILL_DETECTED: (
        "lower VGPR usage via the waves_per_eu hint, or reorder the compute "
        "graph to shorten live ranges"
    ),
    DecisionBottleneckClass.REGISTER_PRESSURE_HIGH: (
        "minimize live variables, move per-thread temp arrays to LDS, or set "
        "__launch_bounds__(block, min_blocks_per_cu)"
    ),
    DecisionBottleneckClass.LDS_PRESSURE_HIGH: (
        "reduce per-block LDS allocation or trim tile size"
    ),
}

_DYNAMIC_LIMITATION = (
    "static occupancy derivation is unreliable under dynamic register "
    "allocation; confirm via runtime profiling"
)


def _ratio(used: int | None, limit: int | None) -> float | None:
    """Return ``used / limit`` or ``None`` when either side is unset or invalid."""

    if used is None or limit is None or limit <= 0:
        return None
    return used / limit


def _hint_identity(
    footprint: StaticResourceFootprint,
) -> DecisionHintIdentity | None:
    """Mirror the footprint identity into a per-hint provenance block."""

    ident = footprint.identity
    if ident is None:
        return None
    return DecisionHintIdentity(
        artifact_id=ident.artifact_id,
        extractor_tool_id=ident.extractor_tool_id,
        source_sha256=ident.source_sha256,
        generated_at=ident.generated_at,
    )


def _occupancy_low(footprint: StaticResourceFootprint, budget: ArchIsaBudget) -> bool:
    """Whether the reported occupancy is below 50% of the architectural ceiling."""

    occ = footprint.occupancy_estimate_waves_per_cu
    ceiling = budget.waves_per_cu_max
    if occ is None or ceiling is None or ceiling <= 0:
        return False
    return occ < ceiling * _OCCUPANCY_LOW_RATIO


def _make_hint(
    *,
    bottleneck_class: DecisionBottleneckClass,
    confidence: DecisionConfidence,
    message: str,
    architecture: str | None,
    footprint: StaticResourceFootprint,
    limitations: Sequence[str] = (),
    evidence_refs: Sequence[str] = (),
) -> DecisionHint:
    return DecisionHint(
        bottleneck_class=bottleneck_class,
        confidence=confidence,
        message=message,
        recommendation=_RECOMMENDATIONS.get(bottleneck_class),
        architecture=architecture,
        identity=_hint_identity(footprint),
        limitations=list(limitations),
        evidence_refs=list(evidence_refs),
    )


def _derive_for_footprint(
    footprint: StaticResourceFootprint,
    budget: ArchIsaBudget | None,
) -> list[DecisionHint]:
    hints: list[DecisionHint] = []
    arch = budget.architecture if budget is not None else None
    occ_low = _occupancy_low(footprint, budget) if budget is not None else False

    # SPILL_DETECTED -- deterministic, highest confidence (research §8.3).
    scratch = footprint.scratch_bytes
    if footprint.spill_detected is True or (scratch is not None and scratch > 0):
        size = f"{scratch} B" if scratch is not None else "unknown size"
        hints.append(
            _make_hint(
                bottleneck_class=DecisionBottleneckClass.SPILL_DETECTED,
                confidence=DecisionConfidence.INFERRED_HIGH,
                message=f"register spill to scratch memory detected ({size})",
                architecture=arch,
                footprint=footprint,
                evidence_refs=["footprint.scratch_bytes"],
            )
        )

    if budget is None:
        return hints

    # REGISTER_PRESSURE_HIGH -- vgpr_limit is the architected addressing limit
    # (research §11.2); used here as the static pressure reference.
    vgpr_ratio = _ratio(footprint.vgpr_used, budget.vgpr_limit)
    if vgpr_ratio is not None and vgpr_ratio >= _REGISTER_PRESSURE_RATIO:
        confidence = (
            DecisionConfidence.INFERRED_HIGH
            if occ_low or vgpr_ratio >= 1.0
            else DecisionConfidence.INFERRED_MEDIUM
        )
        hints.append(
            _make_hint(
                bottleneck_class=DecisionBottleneckClass.REGISTER_PRESSURE_HIGH,
                confidence=confidence,
                message=(
                    f"VGPR usage {footprint.vgpr_used} >= "
                    f"{int(_REGISTER_PRESSURE_RATIO * 100)}% of architected "
                    f"limit {budget.vgpr_limit}"
                ),
                architecture=arch,
                footprint=footprint,
                evidence_refs=["footprint.vgpr_used", "budget.vgpr_limit"],
            )
        )

    # LDS_PRESSURE_HIGH -- per-workgroup LDS budget.
    lds_ratio = _ratio(footprint.lds_bytes, budget.lds_per_workgroup_bytes)
    if lds_ratio is not None and lds_ratio >= _LDS_PRESSURE_RATIO:
        confidence = (
            DecisionConfidence.INFERRED_HIGH
            if occ_low or lds_ratio >= 1.0
            else DecisionConfidence.INFERRED_MEDIUM
        )
        hints.append(
            _make_hint(
                bottleneck_class=DecisionBottleneckClass.LDS_PRESSURE_HIGH,
                confidence=confidence,
                message=(
                    f"LDS allocation {footprint.lds_bytes} B >= "
                    f"{int(_LDS_PRESSURE_RATIO * 100)}% of per-workgroup budget "
                    f"{budget.lds_per_workgroup_bytes} B"
                ),
                architecture=arch,
                footprint=footprint,
                evidence_refs=[
                    "footprint.lds_bytes",
                    "budget.lds_per_workgroup_bytes",
                ],
            )
        )

    return hints


def derive_decision_hints(
    footprints: Sequence[StaticResourceFootprint],
    budget: ArchIsaBudget | None = None,
) -> list[DecisionHint]:
    """Derive Layer R decision hints from static footprints and an arch budget.

    Returns a flat list of hints (one footprint may yield several). Dynamic
    register-allocation budgets emit a single low-confidence note per footprint
    instead of a derived pressure signal. Never promotes unknown values: missing
    fields simply produce no hint.
    """

    if not footprints:
        return []

    # Dynamic register allocation (RDNA4+) defeats static occupancy derivation.
    if budget is not None and budget.register_allocation_model == "dynamic":
        arch = budget.architecture
        return [
            _make_hint(
                bottleneck_class=DecisionBottleneckClass.REGISTER_PRESSURE_HIGH,
                confidence=DecisionConfidence.INFERRED_LOW,
                message=(
                    f"register pressure cannot be statically derived on {arch} "
                    "(dynamic register allocation)"
                ),
                architecture=arch,
                footprint=fp,
                limitations=[_DYNAMIC_LIMITATION],
                evidence_refs=["budget.register_allocation_model"],
            )
            for fp in footprints
        ]

    hints: list[DecisionHint] = []
    for fp in footprints:
        hints.extend(_derive_for_footprint(fp, budget))
    return hints
