# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Closed correctness-check implementations for workload outputs."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from sol_execbench.core.bench.correctness import (
    _sanity_check_view,
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
        out_f = _sanity_check_view(output_chunk).float()
        ref_f = _sanity_check_view(reference_chunk).float()
        max_abs = max(max_abs, float((out_f - ref_f).abs().max().item()))
        max_ref = max(max_ref, float(ref_f.abs().max().item()))
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
    left_view = (
        output.contiguous().view(torch.uint8)
        if output.dtype == torch.float4_e2m1fn_x2
        else output
    )
    right_view = (
        reference.contiguous().view(torch.uint8)
        if reference.dtype == torch.float4_e2m1fn_x2
        else reference
    )
    mismatches = sum(
        int((left != right).sum().item())
        for left, right in zip(
            _tensor_chunks(left_view),
            _tensor_chunks(right_view),
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
    num_experts = int(scores.shape[-1])
    if topk > num_experts or not bool(torch.isfinite(scores).all().item()):
        return 1, 0.0
    sorted_scores = scores.sort(descending=True).values
    cutoff = float(sorted_scores[topk - 1].item())
    ref_id_list = [int(item) for item in reference_ids.tolist()]
    out_id_list = [int(item) for item in output_ids.tolist()]
    reference_set = set(ref_id_list)
    output_set = set(out_id_list)
    in_bounds = all(0 <= item < num_experts for item in out_id_list)
    mismatch = int(
        len(out_id_list) != topk or len(output_set) != topk or not in_bounds
    )
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
    positions = {item: index for index, item in enumerate(ref_id_list)}
    weight_error = 0.0
    for index, item in enumerate(out_id_list):
        weight = float(output_weights[index].item())
        if not math.isfinite(weight):
            mismatch = 1
            continue
        if item in positions:
            expected_weight = float(reference_weights[positions[item]].item())
            if not math.isfinite(expected_weight):
                mismatch = 1
                continue
            weight_error = max(weight_error, abs(weight - expected_weight))
        elif not mismatch:
            weight_error = max(weight_error, abs(weight - cutoff))
    return mismatch, weight_error


def _topk_result(
    check: TopKRoutingCheck,
    input_map: dict[str, object],
    reference_map: dict[str, torch.Tensor],
    output_map: dict[str, torch.Tensor],
    round_index: int,
) -> tuple[TopKRoutingCheckResult, Correctness]:
    gating = torch.as_tensor(input_map[check.gating_input]).float()
    scores = torch.softmax(gating, dim=-1)
    if check.bias_input is not None:
        scores = scores + torch.as_tensor(input_map[check.bias_input]).float()
    reference_ids = (
        reference_map[check.ids_output].cpu().reshape(-1, check.topk)
    )
    reference_weights = (
        reference_map[check.weights_output]
        .float()
        .cpu()
        .reshape(-1, check.topk)
    )
    output_ids = output_map[check.ids_output].cpu().reshape(-1, check.topk)
    output_weights = (
        output_map[check.weights_output].float().cpu().reshape(-1, check.topk)
    )
    score_rows = scores.cpu().reshape(-1, scores.shape[-1])
    has_nan = bool(torch.isnan(output_weights).any().item()) or bool(
        torch.isnan(reference_weights).any().item()
    )
    has_inf = bool(torch.isinf(output_weights).any().item()) or bool(
        torch.isinf(reference_weights).any().item()
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
        for index in range(reference_ids.shape[0])
    ]
    mismatch_ratio = sum(item[0] for item in metrics) / max(len(metrics), 1)
    max_weight_error = max((item[1] for item in metrics), default=0.0)
    passed = (
        not has_nan
        and not has_inf
        and mismatch_ratio <= check.max_mismatch_ratio
        and max_weight_error <= check.weight_atol
    )
    return (
        TopKRoutingCheckResult(
            ids_output=check.ids_output,
            weights_output=check.weights_output,
            round_index=round_index,
            passed=passed,
            genuine_mismatch_ratio=mismatch_ratio,
            max_matched_weight_error=max_weight_error,
        ),
        Correctness(
            max_absolute_error=max_weight_error,
            has_nan=has_nan,
            has_inf=has_inf and not has_nan,
        ),
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
        return _topk_result(
            check, input_map, reference_map, output_map, round_index
        )
    else:  # pragma: no cover - closed schema union
        raise TypeError(f"unsupported check {type(check).__name__}")
    return result, Correctness()


__all__ = ["compare_output_checks"]
