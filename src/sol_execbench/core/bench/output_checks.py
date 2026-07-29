# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Closed correctness-check implementations for workload outputs."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from sol_execbench.core.bench.correctness import (
    _tensor_chunks,
    check_tensor_sanity,
    compute_error_stats,
)
from sol_execbench.core.data.definition import Definition
from sol_execbench.core.data.trace import (
    CodeDistanceCheckResult,
    Correctness,
    ExactCheckResult,
    NumericCheckResult,
    OutputCheckResult,
    TopKRoutingCheckResult,
)
from sol_execbench.core.data.workload import (
    CodeDistanceCheck,
    CodeDistanceMode,
    ExactCheck,
    NumericCheck,
    NumericCheckMode,
    TopKRoutingCheck,
    Workload,
)


def _numeric_elementwise(
    check: NumericCheck,
    output: torch.Tensor,
    reference: torch.Tensor,
    round_index: int,
) -> tuple[NumericCheckResult, Correctness]:
    metrics, exceeds = compute_error_stats(output, reference, check)
    matched_ratio = float((metrics.extra or {}).get("matched_ratio", 0.0))
    result = NumericCheckResult(
        output=check.output,
        round_index=round_index,
        passed=not exceeds,
        max_relative_error=metrics.max_relative_error,
        max_absolute_error=metrics.max_absolute_error,
        matched_ratio=matched_ratio,
        has_nan=metrics.has_nan,
        has_inf=metrics.has_inf,
    )
    return result, metrics


def _normalized_max_metrics(
    output: torch.Tensor,
    reference: torch.Tensor,
    check: NumericCheck,
) -> tuple[float, float, bool, bool]:
    sanity = check_tensor_sanity(
        output,
        reference,
        allow_negative_inf=check.allow_negative_inf,
    )
    if sanity is not None:
        return (
            sanity.max_absolute_error,
            sanity.max_relative_error,
            sanity.has_nan,
            sanity.has_inf,
        )
    max_abs = 0.0
    max_ref = 0.0
    for output_chunk, reference_chunk in zip(
        _tensor_chunks(output),
        _tensor_chunks(reference),
        strict=True,
    ):
        max_abs = max(
            max_abs,
            float(
                (output_chunk.float() - reference_chunk.float())
                .abs()
                .max()
                .item()
            ),
        )
        max_ref = max(
            max_ref, float(reference_chunk.float().abs().max().item())
        )
    return (
        max_abs,
        max_abs / (max_ref + check.denominator_epsilon),
        False,
        False,
    )


def _numeric_normalized_max(
    check: NumericCheck,
    output: torch.Tensor,
    reference: torch.Tensor,
    round_index: int,
) -> tuple[NumericCheckResult, Correctness]:
    max_abs, ratio, has_nan, has_inf = _normalized_max_metrics(
        output,
        reference,
        check,
    )
    exceeds = has_nan or has_inf or ratio > check.max_rtol
    if check.max_error_cap is not None and max_abs > check.max_error_cap:
        exceeds = True
    metrics = Correctness(
        max_absolute_error=max_abs,
        max_relative_error=ratio,
        has_nan=has_nan,
        has_inf=has_inf,
    )
    return (
        NumericCheckResult(
            output=check.output,
            round_index=round_index,
            passed=not exceeds,
            max_relative_error=ratio,
            max_absolute_error=max_abs,
            matched_ratio=0.0 if exceeds else 1.0,
            has_nan=has_nan,
            has_inf=has_inf,
        ),
        metrics,
    )


def _exact_result(
    check: ExactCheck,
    output: torch.Tensor,
    reference: torch.Tensor,
    round_index: int,
) -> ExactCheckResult:
    mismatches = sum(
        int((left != right).sum().item())
        for left, right in zip(
            _tensor_chunks(output),
            _tensor_chunks(reference),
            strict=True,
        )
    )
    return ExactCheckResult(
        output=check.output,
        round_index=round_index,
        passed=mismatches == 0,
        mismatched_elements=mismatches,
        total_elements=output.numel(),
    )


def _code_values(
    tensor: torch.Tensor,
    mode: CodeDistanceMode,
) -> torch.Tensor:
    if mode is CodeDistanceMode.RAW_BITS:
        return tensor.contiguous().view(torch.uint8).to(torch.int32)
    return tensor.to(torch.int64)


def _code_distance_result(
    check: CodeDistanceCheck,
    output: torch.Tensor,
    reference: torch.Tensor,
    round_index: int,
) -> CodeDistanceCheckResult:
    output_codes = _code_values(output, check.mode)
    reference_codes = _code_values(reference, check.mode)
    total = output_codes.numel()
    matched = 0
    max_distance = 0
    for left, right in zip(
        _tensor_chunks(output_codes),
        _tensor_chunks(reference_codes),
        strict=True,
    ):
        distance = (left - right).abs()
        max_distance = max(max_distance, int(distance.max().item()))
        matched += int((distance <= check.max_distance).sum().item())
    ratio = 1.0 if total == 0 else matched / total
    return CodeDistanceCheckResult(
        output=check.output,
        round_index=round_index,
        passed=ratio >= check.required_matched_ratio,
        max_distance=max_distance,
        matched_ratio=ratio,
    )


def _routing_token_metrics(
    reference_ids: torch.Tensor,
    reference_weights: torch.Tensor,
    output_ids: torch.Tensor,
    output_weights: torch.Tensor,
    scores: torch.Tensor,
    topk: int,
    tie_atol: float,
) -> tuple[int, float]:
    sorted_scores = scores.sort(descending=True).values
    cutoff = float(sorted_scores[topk - 1].item())
    reference_set = set(reference_ids.tolist())
    output_set = set(output_ids.tolist())
    mismatch = int(len(output_set) != topk)
    if not mismatch and output_set != reference_set:
        extra = output_set - reference_set
        missing = reference_set - output_set
        tied = all(
            float(scores[item].item()) >= cutoff - tie_atol for item in extra
        )
        tied &= all(
            float(scores[item].item()) <= cutoff + tie_atol for item in missing
        )
        mismatch = int(not tied)
    positions = {
        int(item): index for index, item in enumerate(reference_ids.tolist())
    }
    weight_error = max(
        (
            abs(
                float(output_weights[index].item())
                - float(reference_weights[positions[int(item)]].item())
            )
            for index, item in enumerate(output_ids.tolist())
            if int(item) in positions
        ),
        default=0.0,
    )
    return mismatch, weight_error


def _topk_result(
    check: TopKRoutingCheck,
    input_map: dict[str, object],
    reference_map: dict[str, torch.Tensor],
    output_map: dict[str, torch.Tensor],
    round_index: int,
) -> TopKRoutingCheckResult:
    gating = torch.as_tensor(input_map[check.gating_input]).float()
    scores = torch.softmax(gating, dim=-1)
    if check.bias_input is not None:
        scores = scores + torch.as_tensor(input_map[check.bias_input]).float()
    tensors = (
        reference_map[check.ids_output].cpu(),
        reference_map[check.weights_output].float().cpu(),
        output_map[check.ids_output].cpu(),
        output_map[check.weights_output].float().cpu(),
        scores.cpu(),
    )
    reference_ids, reference_weights, output_ids, output_weights, score_rows = (
        tensors
    )
    metrics = [
        _routing_token_metrics(
            reference_ids[index],
            reference_weights[index],
            output_ids[index],
            output_weights[index],
            score_rows[index],
            check.topk,
            check.tie_atol,
        )
        for index in range(tensors[0].shape[0])
    ]
    mismatch_ratio = sum(item[0] for item in metrics) / max(len(metrics), 1)
    max_weight_error = max((item[1] for item in metrics), default=0.0)
    return TopKRoutingCheckResult(
        ids_output=check.ids_output,
        weights_output=check.weights_output,
        round_index=round_index,
        passed=(
            mismatch_ratio <= check.max_mismatch_ratio
            and max_weight_error <= check.weight_atol
        ),
        genuine_mismatch_ratio=mismatch_ratio,
        max_matched_weight_error=max_weight_error,
    )


def compare_output_checks(
    definition: Definition,
    workload: Workload,
    inputs: Sequence[object],
    reference_outputs: Sequence[torch.Tensor],
    user_outputs: Sequence[torch.Tensor],
    round_index: int,
) -> tuple[Correctness, bool]:
    """Execute every configured output check for one correctness round."""
    input_map = (
        dict(zip(definition.inputs, inputs, strict=True))
        if any(isinstance(check, TopKRoutingCheck) for check in workload.checks)
        else {}
    )
    reference_map = dict(
        zip(definition.outputs, reference_outputs, strict=True)
    )
    output_map = dict(zip(definition.outputs, user_outputs, strict=True))
    results: list[OutputCheckResult] = []
    aggregate = Correctness()
    for check in workload.checks:
        result, metrics = _run_check(
            check,
            input_map,
            reference_map,
            output_map,
            round_index,
        )
        results.append(result)
        aggregate.max_absolute_error = max(
            aggregate.max_absolute_error,
            metrics.max_absolute_error,
        )
        aggregate.max_relative_error = max(
            aggregate.max_relative_error,
            metrics.max_relative_error,
        )
        aggregate.has_nan |= metrics.has_nan
        aggregate.has_inf |= metrics.has_inf
    aggregate.check_results = results
    return aggregate, any(not result.passed for result in results)


def _run_check(
    check: object,
    input_map: dict[str, object],
    reference_map: dict[str, torch.Tensor],
    output_map: dict[str, torch.Tensor],
    round_index: int,
) -> tuple[OutputCheckResult, Correctness]:
    if isinstance(check, NumericCheck):
        function = (
            _numeric_elementwise
            if check.mode is NumericCheckMode.ELEMENTWISE
            else _numeric_normalized_max
        )
        return function(
            check,
            output_map[check.output],
            reference_map[check.output],
            round_index,
        )
    if isinstance(check, ExactCheck):
        result = _exact_result(
            check,
            output_map[check.output],
            reference_map[check.output],
            round_index,
        )
    elif isinstance(check, CodeDistanceCheck):
        result = _code_distance_result(
            check,
            output_map[check.output],
            reference_map[check.output],
            round_index,
        )
    elif isinstance(check, TopKRoutingCheck):
        result = _topk_result(
            check, input_map, reference_map, output_map, round_index
        )
    else:  # pragma: no cover - closed schema union
        raise TypeError(f"unsupported check {type(check).__name__}")
    return result, Correctness()


__all__ = ["compare_output_checks"]
