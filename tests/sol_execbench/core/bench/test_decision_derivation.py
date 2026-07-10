# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""CPU-safe tests for Layer R decision derivation."""

from __future__ import annotations

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
    artifact_id="k0", extractor_tool_id="roc-objdump"
)


def _fp(**kw):
    return StaticResourceFootprint(**kw)


def _cls(hints, value):
    return [h for h in hints if h.bottleneck_class.value == value]


def test_spill_detected_is_inferred_high():
    hints = derive_decision_hints(
        [_fp(scratch_bytes=1024, spill_detected=True, identity=IDENTITY)], GFX942
    )
    spill = _cls(hints, "spill_detected")
    assert spill and spill[0].confidence.value == "inferred_high"
    assert spill[0].identity.artifact_id == "k0"


def test_register_pressure_medium_under_ratio():
    hints = derive_decision_hints([_fp(vgpr_used=250, scratch_bytes=0)], GFX942)
    reg = _cls(hints, "register_pressure_high")
    assert reg and reg[0].confidence.value == "inferred_medium"


def test_register_pressure_high_at_limit():
    hints = derive_decision_hints([_fp(vgpr_used=256)], GFX942)
    assert _cls(hints, "register_pressure_high")[0].confidence.value == "inferred_high"


def test_lds_pressure():
    hints = derive_decision_hints([_fp(lds_bytes=60000)], GFX942)
    assert _cls(hints, "lds_pressure_high")


def test_no_pressure_emits_nothing():
    assert (
        derive_decision_hints(
            [_fp(vgpr_used=20, lds_bytes=1024, scratch_bytes=0)], GFX942
        )
        == []
    )


def test_budget_none_emits_only_spill():
    hints = derive_decision_hints(
        [_fp(vgpr_used=250, scratch_bytes=512, spill_detected=True)], None
    )
    assert len(hints) == 1
    assert hints[0].bottleneck_class.value == "spill_detected"


def test_dynamic_budget_emits_no_layer_r_hints():
    dynamic = GFX942.model_copy(
        update={"register_allocation_model": "dynamic", "architecture": "gfx1200"}
    )
    # Dynamic arch: no Layer R hint (no detected bottleneck); the limitation is
    # recorded at the sidecar level, not as a misleading bottleneck hint.
    assert derive_decision_hints([_fp(vgpr_used=40)], dynamic) == []


def test_dynamic_budget_sidecar_carries_limitation():
    from sol_execbench.core.bench.decision.builder import build_decision_sidecar

    dynamic = GFX942.model_copy(
        update={"register_allocation_model": "dynamic", "architecture": "gfx1200"}
    )
    sidecar = build_decision_sidecar(footprints=[_fp(vgpr_used=40)], budget=dynamic)
    assert sidecar.hints == []
    assert any("dynamic" in lim.lower() for lim in sidecar.limitations)


def test_empty_footprints_returns_empty():
    assert derive_decision_hints([], GFX942) == []


def test_never_promotes_unknown():
    # All-None footprint -> no hint is speculated.
    assert derive_decision_hints([_fp()], GFX942) == []
