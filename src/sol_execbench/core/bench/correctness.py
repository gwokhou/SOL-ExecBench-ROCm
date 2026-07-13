# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Correctness computation utilities."""

from __future__ import annotations

import math
import random
from typing import Optional, Tuple

import torch

from sol_execbench.core.data.trace import EvaluationStatus
from sol_execbench.core.data.trace import Correctness
from sol_execbench.core.data.workload import ToleranceSpec


_CORRECTNESS_CHUNK_ELEMENTS = 1 << 20


def _tensor_chunks(tensor: torch.Tensor):
    """Yield bounded flat views so correctness checks do not scale VRAM overhead."""
    return tensor.reshape(-1).split(_CORRECTNESS_CHUNK_ELEMENTS)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility across Python, CPU, and GPU backends."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def check_output_shape_dtype(
    reference_outputs: list[torch.Tensor],
    user_outputs: list[torch.Tensor],
) -> EvaluationStatus | None:
    """Return the first structural output mismatch status, if any."""
    for ref_out, user_out in zip(reference_outputs, user_outputs):
        if ref_out.shape != user_out.shape:
            return EvaluationStatus.INCORRECT_SHAPE
        if ref_out.dtype != user_out.dtype:
            return EvaluationStatus.INCORRECT_DTYPE
    return None


def check_tensor_sanity(
    sol_tensor: torch.Tensor,
    ref_tensor: torch.Tensor,
    allow_negative_inf: bool = False,
) -> Optional[Correctness]:
    """Check for non-finite values and all-zeros output.

    Returns a ``Correctness`` describing the failure when either tensor
    contains inf/nan values, or the solution is all zeros while the
    reference is not.  Returns ``None`` when both tensors look sane.

    Inf and NaN are always treated as incorrect, even when both tensors
    have the same non-finite value at a given position — unless
    *allow_negative_inf* is True, in which case positions where **both**
    tensors are -inf are tolerated.
    """
    has_nonfinite = False
    has_nan = False
    ref_nonzero = False
    sol_nonzero = False
    ref_norm_squared = 0.0
    for sol_chunk, ref_chunk in zip(
        _tensor_chunks(sol_tensor), _tensor_chunks(ref_tensor), strict=True
    ):
        ref_nonfinite = ~torch.isfinite(ref_chunk)
        sol_nonfinite = ~torch.isfinite(sol_chunk)
        if allow_negative_inf:
            both_neg_inf = (ref_chunk == float("-inf")) & (sol_chunk == float("-inf"))
            ref_nonfinite &= ~both_neg_inf
            sol_nonfinite &= ~both_neg_inf
        if bool(ref_nonfinite.any().item()) or bool(sol_nonfinite.any().item()):
            has_nonfinite = True
            has_nan = has_nan or bool(torch.isnan(sol_chunk).any().item())
            has_nan = has_nan or bool(torch.isnan(ref_chunk).any().item())
        ref_nonzero = ref_nonzero or bool(torch.count_nonzero(ref_chunk).item())
        sol_nonzero = sol_nonzero or bool(torch.count_nonzero(sol_chunk).item())
        chunk_norm = float(torch.linalg.vector_norm(ref_chunk.float()).item())
        ref_norm_squared += chunk_norm * chunk_norm

    if has_nonfinite:
        # has_inf is True only when there are Inf values but NO NaN values.
        # NaN takes priority because it's a stricter failure mode (Inf can
        # result from overflow, but NaN indicates undefined computation).
        return Correctness(has_nan=has_nan, has_inf=not has_nan)

    # Non-zero output check: if reference has non-trivial values
    # but solution is all zeros, fail immediately.
    if ref_nonzero and not sol_nonzero:
        abs_err = math.sqrt(ref_norm_squared)
        return Correctness(
            max_absolute_error=abs_err,
            max_relative_error=abs_err,
        )

    return None


def compute_error_stats(
    output: torch.Tensor, reference: torch.Tensor, tolerance: ToleranceSpec
) -> Tuple[Correctness, bool]:
    """Compute numerical error between *output* and *reference*.

    Returns ``(correctness, exceeds)`` where *correctness* is a
    :class:`Correctness` carrying error metrics (and ``has_nan`` /
    ``has_inf`` flags when non-finite values are detected), and *exceeds*
    is ``True`` when the tolerance is violated.
    """
    allow_neg_inf = tolerance.allow_negative_inf

    # Automatically fail on infs/nans in either tensor even if they're in the same position.
    infs_nans = check_tensor_sanity(output, reference, allow_negative_inf=allow_neg_inf)
    if infs_nans is not None:
        return infs_nans, True

    total_elements = 0
    exceeds_count = 0
    max_abs = 0.0
    max_rel = 0.0
    for output_chunk, reference_chunk in zip(
        _tensor_chunks(output), _tensor_chunks(reference), strict=True
    ):
        x = output_chunk.float()
        y = reference_chunk.float()
        if allow_neg_inf:
            finite_mask = ~((x == float("-inf")) & (y == float("-inf")))
            x = x[finite_mask]
            y = y[finite_mask]
        if not x.numel():
            continue
        abs_error = torch.abs(x - y)
        total_elements += abs_error.numel()
        max_abs = max(max_abs, float(abs_error.max().item()))
        tol_bound = tolerance.max_atol + tolerance.max_rtol * torch.abs(y)
        exceeds_tol_mask = (abs_error > tol_bound) | ~torch.isfinite(abs_error)
        exceeds_count += int(exceeds_tol_mask.sum().item())
        rel_error = abs_error / torch.clamp(torch.abs(y), min=tolerance.max_atol)
        max_rel = max(max_rel, float(rel_error.max().item()))

    if total_elements == 0:
        return Correctness(), False
    matched_ratio = 1.0 - (exceeds_count / float(total_elements))
    matched_ratio = max(0.0, min(1.0, matched_ratio))

    # Hard ceiling on max absolute error for library-style kernels.
    # Prevents accepting solutions where most elements match but rare outliers
    # have arbitrarily large errors.
    exceeds_tol = matched_ratio < tolerance.required_matched_ratio
    if tolerance.max_error_cap is not None and max_abs > tolerance.max_error_cap:
        exceeds_tol = True

    return Correctness(
        max_absolute_error=max_abs, max_relative_error=max_rel
    ), exceeds_tol
