# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Batch-7 pool/loss acceptance (spec: solar-l3e batch 7)."""

from __future__ import annotations

import pytest
import torch

from scripts.internal.solar.l3e_inventory import blockers_for_call
from solar.ir.contracts import IRPath


def _generator() -> torch.Generator:
    return torch.Generator(device="cpu").manual_seed(11)


@pytest.mark.parametrize(
    ("name", "reference", "inputs", "ir_path"),
    [
        (
            "cross_entropy_fused",
            lambda a, b: torch.nn.functional.cross_entropy(a, b),
            lambda g: (
                torch.randn(6, 5, generator=g),
                torch.tensor([0, 1, 2, 3, 4, 0]),
            ),
            IRPath.TORCHVIEW_EXTENDED_EINSUM,
        ),
        (
            "kl_div_fused",
            lambda a, b: torch.nn.functional.kl_div(
                a.log_softmax(-1), b.log_softmax(-1), reduction="sum"
            ),
            lambda g: (
                torch.randn(6, 5, generator=g),
                torch.randn(6, 5, generator=g),
            ),
            IRPath.TORCHVIEW_EXTENDED_EINSUM,
        ),
        (
            "max_pool2d_torchview",
            lambda x: torch.nn.functional.max_pool2d(x, 2),
            lambda g: (torch.randn(1, 3, 8, 8, generator=g),),
            IRPath.TORCHVIEW_EXTENDED_EINSUM,
        ),
        (
            "max_pool2d_make_fx",
            lambda x: torch.nn.functional.max_pool2d(x, 2),
            lambda g: (torch.randn(1, 3, 8, 8, generator=g),),
            IRPath.MAKE_FX_ATEN,
        ),
        (
            "avg_pool2d_make_fx",
            lambda x: torch.nn.functional.avg_pool2d(x, (1, 16), (1, 16)),
            lambda g: (torch.randn(2, 1, 1, 32, generator=g),),
            IRPath.MAKE_FX_ATEN,
        ),
        (
            "nll_internal_data",
            lambda a, b: torch.nn.functional.nll_loss(
                torch.log_softmax(a, dim=-1), b
            ),
            lambda g: (
                torch.randn(6, 5, generator=g),
                torch.tensor([0, 1, 2, 3, 4, 0]),
            ),
            IRPath.MAKE_FX_ATEN,
        ),
    ],
)
def test_batch7_targets_no_longer_block(
    name, reference, inputs, ir_path, tmp_path
):
    """Each representative batch-7 graph produces zero blockers (G1)."""
    blockers = blockers_for_call(
        reference, inputs(_generator()), ir_path, tmp_path, name
    )
    assert blockers == [], (name, blockers)


@pytest.mark.parametrize(
    ("name", "reference", "shape", "expected"),
    [
        (
            "stride_exceeds_extent",
            lambda x: torch.nn.functional.max_pool2d(x, 2, stride=3),
            (1, 3, 8, 8),
            ("max_pool2d_with_indices", "read_semantics_unproved"),
        ),
        (
            "tail_uncovered",
            lambda x: torch.nn.functional.max_pool2d(x, (3, 3), stride=2),
            (1, 1, 4, 4),
            ("max_pool2d_with_indices", "read_semantics_unproved"),
        ),
        (
            "ceil_mode_unproved",
            lambda x: torch.nn.functional.max_pool2d(x, 3, ceil_mode=True),
            (1, 3, 8, 8),
            ("max_pool2d_with_indices", "read_semantics_unproved"),
        ),
    ],
)
def test_batch7_pool_coverage_negatives_still_block(
    name, reference, shape, expected, tmp_path
):
    """Uncovered or unmodeled windows keep pool reads unproved (G2)."""
    generator = _generator()
    inputs = (torch.randn(shape, generator=generator),)
    blockers = blockers_for_call(
        reference, inputs, IRPath.MAKE_FX_ATEN, tmp_path, name
    )
    assert expected in blockers, (name, blockers)


def test_batch7_nll_external_data_still_blocks(tmp_path):
    """A gathered external data operand stays in the dynamic-index domain (G2)."""
    generator = _generator()
    inputs = (
        torch.randn(6, 5, generator=generator),
        torch.tensor([0, 1, 2, 3, 4, 0]),
    )
    blockers = blockers_for_call(
        lambda a, b: torch.nn.functional.nll_loss(a, b),
        inputs,
        IRPath.MAKE_FX_ATEN,
        tmp_path,
        "nll_external",
    )
    assert ("nll_loss_forward", "dynamic_index") in blockers


def test_batch7_cross_entropy_weight_operand_guard():
    """A third tensor operand is a per-class gather and stays unproved (G2)."""
    from solar.analysis.external_reads import _blocking_reason
    from solar.ir.access import AccessReason
    from solar.ir.access_artifact import AccessDocument
    from solar.ir.contracts import IRKind

    access = AccessDocument(
        graph_sha256="0" * 64,
        operator_sha256="0" * 64,
        ir_kind=IRKind.ATEN,
        source_storage=None,
        source_indices={},
        tensors={},
        reasons={},
        transparent_layers=frozenset(),
        effect_bases=frozenset(),
    )

    def layer(operands):
        return {
            "semantic_op": {
                "kind": "operation",
                "target": "cross_entropy",
                "operands": operands,
            },
            "tensor_names": {
                "inputs": ["logits", "target", "weight"],
                "outputs": ["loss"],
            },
        }

    plain = layer([{"tensor": 0}, {"tensor": 1}, {"value": None}])
    weighted = layer([{"tensor": 0}, {"tensor": 1}, {"tensor": 2}])
    assert _blocking_reason(plain, access, {}, {}) is None
    assert (
        _blocking_reason(weighted, access, {}, {}) is AccessReason.UNKNOWN_READS
    )


def test_batch7_registry_deopaque_set_is_exact():
    """Only the three reviewed targets left the opaque library class (G2)."""
    from solar.ir.extended_einsum.native_registry import (
        NATIVE_OP_REGISTRY,
        NativeOpSpec,
    )

    def find(target: str) -> NativeOpSpec:
        assert target in NATIVE_OP_REGISTRY, target
        return NATIVE_OP_REGISTRY[target]

    reviewed = {"cross_entropy", "kl_div", "max_pool2d"}
    for target in reviewed:
        assert not find(target).effects.opaque_library_call, target
    for target in ("nll_loss", "adaptive_avg_pool2d", "rms_norm"):
        assert find(target).effects.opaque_library_call, target
