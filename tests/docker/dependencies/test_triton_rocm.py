# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
import triton
import triton.language as tl


def test_triton_rocm_import_and_backend():
    assert triton is not None
    assert tl is not None
    assert hasattr(triton, "__version__")
    assert torch.version.hip is not None, "PyTorch ROCm backend is required for Triton ROCm coverage"
    assert torch.cuda.is_available(), "No ROCm GPU visible through PyTorch"
