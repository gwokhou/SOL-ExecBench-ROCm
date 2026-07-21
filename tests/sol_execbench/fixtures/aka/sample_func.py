# SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Minimal fixture mirroring the AMD AgentKernelArena torch2* functional
# reference layout, for parser unit tests. Derived from the public AKA
# torch2hip/gpumode/3267_SimpleMatmulModule task shape (Apache-2.0).

import torch


def module_fn(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Performs matrix multiply of a with (b + b)."""
    return torch.matmul(a, b + b)


def get_inputs():
    for shapes in [([4, 4], [4, 4]), ([16, 32], [32, 16])]:
        yield [torch.rand(*shapes[0]), torch.rand(*shapes[1])]


def get_init_inputs():
    return [[], {}]
