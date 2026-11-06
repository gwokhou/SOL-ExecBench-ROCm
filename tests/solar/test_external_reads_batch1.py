# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Batch-1 external-reads vocabulary acceptance (spec: solar-l3e)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from scripts.internal.solar.l3e_inventory import blockers_for_call
from solar.analysis.external_reads import FULL_OPERAND_READS
from solar.ir.contracts import IRPath

BATCH1_VOCABULARY = frozenset(
    {
        "cat",
        "pow",
        "sin",
        "cos",
        "log2",
        "floor",
        "exp2",
        "log_softmax",
        "softplus",
        "leaky_relu",
        "bitwise_and",
        "xlogy",
        "masked_fill",
        "expand",
        "repeat",
    }
)


def test_batch1_vocabulary_is_frozen_in_full_operand_reads() -> None:
    """Every batch-1 target joined the full-operand read set (G1)."""
    assert BATCH1_VOCABULARY <= FULL_OPERAND_READS


def _inputs(*shapes: tuple[int, ...]) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device="cpu").manual_seed(11)
    return tuple(torch.randn(shape, generator=generator) for shape in shapes)


@pytest.mark.parametrize(
    ("name", "reference", "shapes"),
    [
        ("cat", lambda a, b: torch.cat((a, b), dim=-1), ((4, 6), (4, 6))),
        ("pow", lambda a, b: a.pow(b), ((4, 6), (4, 6))),
        ("sin", lambda a: torch.sin(a) * 2.0, ((4, 6),)),
        ("cos", lambda a: torch.cos(a) + 1.0, ((4, 6),)),
        ("log_softmax", lambda a: torch.log_softmax(a, dim=-1), ((4, 6),)),
        ("softplus", lambda a: torch.nn.functional.softplus(a), ((4, 6),)),
        ("leaky_relu", lambda a: torch.nn.functional.leaky_relu(a), ((4, 6),)),
        ("bitwise_and", None, ((4, 6),)),
        ("xlogy", lambda a, b: torch.xlogy(a, b), ((4, 6), (4, 6))),
        (
            "masked_fill",
            lambda a, mask: a.masked_fill(mask, 1.0),
            ((4, 6), (4, 6)),
        ),
        (
            "expand",
            lambda a: a.unsqueeze(0).expand(3, -1, -1).sum(),
            ((4, 6),),
        ),
        (
            "repeat",
            lambda a: a.repeat(2, 3).sum(),
            ((4, 6),),
        ),
    ],
)
def test_batch1_targets_no_longer_block(name, reference, shapes, tmp_path):
    """Each representative batch-1 graph produces zero blockers (G1)."""
    generator = torch.Generator(device="cpu").manual_seed(11)
    if name == "bitwise_and":
        inputs = tuple(
            torch.randint(0, 7, shape, generator=generator) for shape in shapes
        )

        def reference(a):
            return a.bitwise_and(3)

    elif name == "masked_fill":
        base = _inputs(*shapes)
        inputs = (base[0], base[1] > 0)
    else:
        inputs = _inputs(*shapes)
    blockers = blockers_for_call(
        reference, inputs, IRPath.MAKE_FX_ATEN, tmp_path, name
    )
    assert blockers == [], (name, blockers)


def test_strided_conv2d_still_blocks(tmp_path: Path) -> None:
    """A strided conv is not a full contraction and must stay unproved (G2)."""
    inputs = (
        torch.randn(2, 3, 8, 8),
        torch.randn(5, 3, 3, 3),
    )

    def reference(x, weight):
        return torch.nn.functional.conv2d(x, weight, stride=2)

    blockers = blockers_for_call(
        reference, inputs, IRPath.MAKE_FX_ATEN, tmp_path, "conv_stride"
    )
    assert ("conv2d", "read_semantics_unproved") in blockers


def test_layer_norm_reads_all_operands(tmp_path: Path) -> None:
    """Batch-3 spec: normalization reads every operand and no longer blocks."""
    inputs = (torch.randn(4, 16),)

    def reference(x):
        return torch.nn.functional.layer_norm(x, (16,))

    blockers = blockers_for_call(
        reference, inputs, IRPath.MAKE_FX_ATEN, tmp_path, "layer_norm"
    )
    # The declared-output alias remains batch-2+ domain; the norm read
    # semantics itself no longer blocks.
    assert ("layer_norm", "read_semantics_unproved") not in blockers


def test_indexed_loss_external_data_still_blocks(tmp_path: Path) -> None:
    """An external gathered nll operand stays dynamic-index (G2, batch 7)."""
    generator = torch.Generator(device="cpu").manual_seed(11)
    inputs = (
        torch.randn(6, 5, generator=generator),
        torch.tensor([0, 1, 2, 3, 4, 0]),
    )

    def reference(logits, targets):
        # Batch-7 atomic update (pre-declared in its spec): batch 1 pinned
        # the decomposed loss as unproved; batch 7 proves the internal-data
        # shape (see test_external_reads_batch7) and this test keeps the
        # still-true negative — a gathered external data operand.
        return torch.nn.functional.nll_loss(logits, targets)

    blockers = blockers_for_call(
        reference, inputs, IRPath.MAKE_FX_ATEN, tmp_path, "nll_loss"
    )
    assert ("nll_loss_forward", "dynamic_index") in blockers
