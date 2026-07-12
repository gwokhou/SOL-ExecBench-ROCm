# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Random input tensor generation behavior."""

from __future__ import annotations

import pytest
import torch

from sol_execbench.core.bench.io import _rand_tensor


class TestRandTensor:
    def test_float32(self) -> None:
        tensor = _rand_tensor([4, 8], torch.float32, torch.device("cpu"))
        assert tensor.shape == (4, 8)
        assert tensor.dtype == torch.float32

    def test_float16(self) -> None:
        assert (
            _rand_tensor([3], torch.float16, torch.device("cpu")).dtype == torch.float16
        )

    def test_bfloat16(self) -> None:
        assert (
            _rand_tensor([2, 2], torch.bfloat16, torch.device("cpu")).dtype
            == torch.bfloat16
        )

    @pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
    def test_float8(self, dtype: torch.dtype) -> None:
        assert _rand_tensor([16], dtype, torch.device("cpu")).dtype == dtype

    def test_bool(self) -> None:
        tensor = _rand_tensor([100], torch.bool, torch.device("cpu"))
        assert tensor.dtype == torch.bool
        assert set(tensor.unique().tolist()).issubset({True, False})

    def test_int8(self) -> None:
        tensor = _rand_tensor([100], torch.int8, torch.device("cpu"))
        assert tensor.min().item() >= -128
        assert tensor.max().item() < 128

    @pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
    def test_integer_dtype(self, dtype: torch.dtype) -> None:
        assert _rand_tensor([50], dtype, torch.device("cpu")).dtype == dtype

    def test_unsupported_dtype(self) -> None:
        with pytest.raises(ValueError, match="Unsupported random dtype"):
            _rand_tensor([4], torch.complex64, torch.device("cpu"))
