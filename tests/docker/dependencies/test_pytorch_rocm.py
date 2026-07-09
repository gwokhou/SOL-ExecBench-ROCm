# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
import pytest

pytestmark = [
    pytest.mark.docker_dependency,
    pytest.mark.requires_rocm,
    pytest.mark.requires_rocm_gpu,
]


def test_torch_rocm_backend_available():
    assert torch.version.hip is not None, "PyTorch is not a ROCm build"
    assert torch.cuda.is_available(), "No ROCm GPU visible through PyTorch"
    x = torch.ones((4,), device="cuda")
    y = x + 1
    torch.testing.assert_close(y, torch.full((4,), 2.0, device="cuda"))
