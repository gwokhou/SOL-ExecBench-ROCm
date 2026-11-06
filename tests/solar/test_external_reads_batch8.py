# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Batch-8 conv window-coverage acceptance (spec: solar-l3e batch 8)."""

from __future__ import annotations

import pytest
import torch

from scripts.internal.solar.l3e_inventory import blockers_for_call
from solar.ir.contracts import IRPath


def _generator() -> torch.Generator:
    return torch.Generator(device="cpu").manual_seed(11)


@pytest.mark.parametrize(
    ("name", "reference", "input_fn", "weight_shape", "groups"),
    [
        (
            "conv_k3s1_dense",
            lambda a, w: torch.nn.functional.conv2d(
                a, w, None, (1, 1), (0, 0), (1, 1), 1
            ),
            lambda g: (2, 8, 32, 32),
            (16, 8, 3, 3),
            1,
        ),
        (
            "conv_k3s1_depthwise_g8",
            lambda a, w: torch.nn.functional.conv2d(
                a, w, None, (1, 1), (0, 0), (1, 1), 8
            ),
            lambda g: (2, 8, 32, 32),
            (8, 1, 3, 3),
            8,
        ),
        (
            "conv_k3s1_tail_exact",
            lambda a, w: torch.nn.functional.conv2d(a, w),
            lambda g: (1, 4, 7, 7),
            (2, 4, 3, 3),
            1,
        ),
    ],
)
def test_batch8_covered_convs_no_longer_block(
    name, reference, input_fn, weight_shape, groups, tmp_path
):
    """Window-covered convs prove their unique-element union read (G1)."""
    generator = _generator()
    inputs = (
        torch.randn(input_fn(generator), generator=generator),
        torch.randn(weight_shape, generator=generator),
    )
    assert groups in (1, weight_shape[0])
    blockers = blockers_for_call(
        reference, inputs, IRPath.MAKE_FX_ATEN, tmp_path, name
    )
    assert blockers == [], (name, blockers)


@pytest.mark.parametrize(
    ("name", "reference", "input_shape", "weight_shape"),
    [
        (
            "stride_gap_uncovered",
            lambda a, w: torch.nn.functional.conv2d(a, w, None, (2, 2)),
            (2, 8, 32, 32),
            (16, 8, 3, 3),
        ),
        (
            "transposed_stays_unproved",
            lambda a, w: torch.nn.functional.conv_transpose2d(a, w),
            (2, 8, 32, 32),
            (8, 16, 3, 3),
        ),
    ],
)
def test_batch8_conv_negatives_still_block(
    name, reference, input_shape, weight_shape, tmp_path
):
    """Uncovered or scatter-shaped convs keep their unproved reason (G2)."""
    generator = _generator()
    inputs = (
        torch.randn(input_shape, generator=generator),
        torch.randn(weight_shape, generator=generator),
    )
    blockers = blockers_for_call(
        reference, inputs, IRPath.MAKE_FX_ATEN, tmp_path, name
    )
    assert ("conv2d", "read_semantics_unproved") in blockers or (
        "conv_transpose2d",
        "read_semantics_unproved",
    ) in blockers


def test_batch8_strided_conv_negative_unchanged(tmp_path):
    """The batch-1 strided-conv negative keeps its exact reason (G2)."""
    generator = _generator()
    inputs = (
        torch.randn(2, 3, 8, 8, generator=generator),
        torch.randn(5, 3, 3, 3, generator=generator),
    )
    blockers = blockers_for_call(
        lambda x, weight: torch.nn.functional.conv2d(x, weight, stride=2),
        inputs,
        IRPath.MAKE_FX_ATEN,
        tmp_path,
        "conv_stride",
    )
    assert ("conv2d", "read_semantics_unproved") in blockers
