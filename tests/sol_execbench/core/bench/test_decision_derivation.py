# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""CPU-safe tests for Layer R decision derivation."""

from __future__ import annotations

from sol_execbench.core.bench.decision.decision_models import (
    DecisionBottleneckClass,
    DecisionConfidence,
)
from sol_execbench.core.bench.decision.derivation import derive_decision_hints
from sol_execbench.core.bench.static_kernel.evidence_models import (
    StaticResourceFootprint,
    StaticResourceFootprintIdentity,
)
from sol_execbench.core.platform.arch_capabilities import (
    load_packaged_arch_capability_budget,
)

GFX942 = load_packaged_arch_capability_budget("gfx942")
IDENTITY = StaticResourceFootprintIdentity(
    artifact_id="k0",
    extractor_tool_id="roc-objdump",
)


def _fp(**kw):
    return StaticResourceFootprint(**kw)


def _cls(hints, value: DecisionBottleneckClass):
    return [h for h in hints if h.bottleneck_class is value]


def test_spill_detected_is_inferred_high():
    hints = derive_decision_hints(
        [_fp(scratch_bytes=1024, spill_detected=True, identity=IDENTITY)],
        GFX942,
    )
    spill = _cls(hints, DecisionBottleneckClass.SPILL_DETECTED)
    assert spill and spill[0].confidence is DecisionConfidence.INFERRED_HIGH
    assert spill[0].identity.artifact_id == "k0"


def test_register_pressure_medium_under_ratio():
    hints = derive_decision_hints([_fp(vgpr_used=250, scratch_bytes=0)], GFX942)
    reg = _cls(hints, DecisionBottleneckClass.REGISTER_PRESSURE_HIGH)
    assert reg and reg[0].confidence is DecisionConfidence.INFERRED_MEDIUM


def test_register_pressure_high_at_limit():
    hints = derive_decision_hints([_fp(vgpr_used=256)], GFX942)
    assert (
        _cls(hints, DecisionBottleneckClass.REGISTER_PRESSURE_HIGH)[
            0
        ].confidence
        is DecisionConfidence.INFERRED_HIGH
    )


def test_lds_pressure():
    hints = derive_decision_hints([_fp(lds_bytes=60000)], GFX942)
    assert _cls(hints, DecisionBottleneckClass.LDS_PRESSURE_HIGH)


def test_no_pressure_emits_nothing():
    assert (
        derive_decision_hints(
            [_fp(vgpr_used=20, lds_bytes=1024, scratch_bytes=0)],
            GFX942,
        )
        == []
    )


def test_budget_none_emits_only_spill():
    hints = derive_decision_hints(
        [_fp(vgpr_used=250, scratch_bytes=512, spill_detected=True)],
        None,
    )
    assert len(hints) == 1
    assert hints[0].bottleneck_class is DecisionBottleneckClass.SPILL_DETECTED


def test_dynamic_budget_emits_no_layer_r_hints():
    dynamic = GFX942.model_copy(
        update={
            "register_allocation_model": "dynamic",
            "architecture": "gfx1200",
        },
    )
    # Dynamic arch: no Layer R hint (no detected bottleneck); the limitation is
    # recorded at the sidecar level, not as a misleading bottleneck hint.
    assert derive_decision_hints([_fp(vgpr_used=40)], dynamic) == []


def test_dynamic_budget_sidecar_carries_limitation():
    from sol_execbench.core.bench.decision.builder import build_decision_sidecar

    dynamic = GFX942.model_copy(
        update={
            "register_allocation_model": "dynamic",
            "architecture": "gfx1200",
        },
    )
    sidecar = build_decision_sidecar(
        footprints=[_fp(vgpr_used=40)],
        budget=dynamic,
    )
    assert sidecar.hints == []
    assert any("dynamic" in lim.lower() for lim in sidecar.limitations)


def test_empty_footprints_returns_empty():
    assert derive_decision_hints([], GFX942) == []


def test_never_promotes_unknown():
    # All-None footprint -> no hint is speculated.
    assert derive_decision_hints([_fp()], GFX942) == []


def test_dynamic_budget_still_emits_spill():
    # Spill is deterministic and arch-agnostic, so a dynamic-allocation budget
    # (gfx1200/RDNA4) must still emit SPILL_DETECTED; only pressure is gated.
    dynamic = GFX942.model_copy(
        update={
            "register_allocation_model": "dynamic",
            "architecture": "gfx1200",
        },
    )
    hints = derive_decision_hints(
        [_fp(vgpr_used=250, scratch_bytes=1024, spill_detected=True)],
        dynamic,
    )
    spill = _cls(hints, DecisionBottleneckClass.SPILL_DETECTED)
    assert spill and spill[0].confidence is DecisionConfidence.INFERRED_HIGH
    # register pressure is suppressed on dynamic budgets even though vgpr(250)
    # would otherwise cross the ratio threshold.
    assert _cls(hints, DecisionBottleneckClass.REGISTER_PRESSURE_HIGH) == []


def test_dynamic_budget_suppresses_pressure_without_spill():
    dynamic = GFX942.model_copy(
        update={
            "register_allocation_model": "dynamic",
            "architecture": "gfx1200",
        },
    )
    # vgpr_used=256 would trigger register_pressure on a static budget.
    assert derive_decision_hints([_fp(vgpr_used=256)], dynamic) == []
