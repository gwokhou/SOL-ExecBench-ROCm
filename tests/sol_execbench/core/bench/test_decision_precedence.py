# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""CPU-safe tests for cross-sidecar runtime precedence."""

from __future__ import annotations

from sol_execbench.core.bench.decision.builder import build_decision_sidecar
from sol_execbench.core.bench.decision.precedence import (
    apply_runtime_precedence,
    runtime_precedence_limitation,
)
from sol_execbench.core.bench.static_kernel.evidence_models import (
    StaticResourceFootprint,
)
from sol_execbench.core.platform.arch_capabilities import (
    load_packaged_arch_capability_budget,
)

GFX942 = load_packaged_arch_capability_budget("gfx942")
_FOOTPRINT = StaticResourceFootprint(scratch_bytes=8, spill_detected=True)


def test_runtime_available_adds_precedence_note():
    sidecar = build_decision_sidecar(footprints=[_FOOTPRINT], budget=GFX942)
    annotated = apply_runtime_precedence(sidecar, runtime_profile_available=True)

    assert runtime_precedence_limitation() in annotated.limitations
    assert annotated.status == sidecar.status  # status unchanged


def test_no_runtime_leaves_sidecar_unchanged():
    sidecar = build_decision_sidecar(footprints=[_FOOTPRINT], budget=GFX942)
    assert apply_runtime_precedence(sidecar, runtime_profile_available=False) is sidecar


def test_unavailable_sidecar_not_annotated():
    sidecar = build_decision_sidecar(footprints=[], budget=GFX942)
    annotated = apply_runtime_precedence(sidecar, runtime_profile_available=True)
    assert runtime_precedence_limitation() not in annotated.limitations


def test_precedence_annotation_is_idempotent():
    sidecar = build_decision_sidecar(footprints=[_FOOTPRINT], budget=GFX942)
    once = apply_runtime_precedence(sidecar, runtime_profile_available=True)
    twice = apply_runtime_precedence(once, runtime_profile_available=True)
    assert once.limitations == twice.limitations
