# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Batch-6 transparent-alias acceptance (spec: solar-l3e batch 6)."""

from __future__ import annotations

import pytest
import torch

from scripts.internal.solar.l3e_inventory import blockers_for_call
from solar.analysis.external_reads import (
    transparent_declared_output_aliases_external,
)
from solar.ir.access import AccessReason, base_access
from solar.ir.access_artifact import AccessDocument
from solar.ir.contracts import IRKind, IRPath


def _inputs(*shapes: tuple[int, ...]) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device="cpu").manual_seed(11)
    return tuple(torch.randn(shape, generator=generator) for shape in shapes)


def test_internal_materialized_declared_alias_no_longer_blocks(tmp_path):
    """A detach over an internally materialized value is internal storage (G1)."""

    def reference(x):
        return torch.nn.functional.gelu(x).detach()

    blockers = blockers_for_call(
        reference, _inputs((4, 6)), IRPath.MAKE_FX_ATEN, tmp_path, "gelu_detach"
    )
    assert blockers == []


def test_internal_materialized_view_output_no_longer_blocks(tmp_path):
    """A view over an internally materialized value is internal storage (G1)."""

    def reference(x):
        return torch.nn.functional.gelu(x).view(2, 12)

    blockers = blockers_for_call(
        reference, _inputs((4, 6)), IRPath.MAKE_FX_ATEN, tmp_path, "gelu_view"
    )
    assert blockers == []


def test_external_detached_output_still_blocks(tmp_path):
    """A detach of a tracked external base keeps the alias unproved (G2)."""

    def reference(x):
        return x.detach()

    blockers = blockers_for_call(
        reference, _inputs((4, 6)), IRPath.MAKE_FX_ATEN, tmp_path, "detach_ext"
    )
    assert blockers == [("<transparent_alias>", "unproven_alias")]


def test_external_view_output_still_blocks(tmp_path):
    """A declared view of a tracked external base keeps the alias unproved (G2)."""

    def reference(x):
        return x.view(2, 12)

    blockers = blockers_for_call(
        reference, _inputs((4, 6)), IRPath.MAKE_FX_ATEN, tmp_path, "view_ext"
    )
    assert blockers == [("<transparent_alias>", "unproven_alias")]


def _access(
    *, tracked: bool, reason: AccessReason
) -> tuple[AccessDocument, dict]:
    tensors = {"src": base_access("x", (4, 6), "fp32")} if tracked else {}
    reasons = {} if tracked else {"src": reason}
    access = AccessDocument(
        graph_sha256="0" * 64,
        operator_sha256="0" * 64,
        ir_kind=IRKind.ATEN,
        source_storage=None,
        source_indices={},
        tensors=tensors,
        reasons=reasons,
        transparent_layers=frozenset({"alias"}),
        effect_bases=frozenset(),
    )
    layer = {
        "tensor_names": {"inputs": ["src"], "outputs": ["out"]},
    }
    return access, layer


@pytest.mark.parametrize(
    ("tracked", "reason", "blocks"),
    [
        (False, AccessReason.MATERIALIZED, False),
        (True, AccessReason.STATIC, True),
        (False, AccessReason.UNSUPPORTED_VIEW, True),
        (False, AccessReason.COPY, True),
        (False, AccessReason.EFFECT, True),
        (False, AccessReason.DYNAMIC_INDEX, True),
    ],
)
def test_declared_alias_gate_decides_from_input_storage(
    tracked, reason, blocks
):
    """Only graph-owned fresh storage proves the alias internal (G1/G2)."""
    access, layer = _access(tracked=tracked, reason=reason)
    assert (
        transparent_declared_output_aliases_external(layer, {"out"}, access)
        is blocks
    )


def test_gate_ignores_undeclared_outputs_and_fails_closed_without_inputs():
    """Undeclared transparent outputs never block; no inputs cannot prove storage."""
    access, layer = _access(tracked=False, reason=AccessReason.MATERIALIZED)
    assert (
        transparent_declared_output_aliases_external(layer, {"other"}, access)
        is False
    )
    empty = {"tensor_names": {"inputs": [], "outputs": ["out"]}}
    assert (
        transparent_declared_output_aliases_external(empty, {"out"}, access)
        is True
    )


def test_gate_requires_every_input_materialized():
    """One unproved co-input keeps the declared alias unproved (G2)."""
    access, layer = _access(tracked=False, reason=AccessReason.MATERIALIZED)
    layer["tensor_names"]["inputs"] = ["src", "peer"]
    mixed = access.model_copy(
        update={
            "reasons": {
                "src": AccessReason.MATERIALIZED,
                "peer": AccessReason.UNSUPPORTED_VIEW,
            }
        }
    )
    assert (
        transparent_declared_output_aliases_external(layer, {"out"}, mixed)
        is True
    )
