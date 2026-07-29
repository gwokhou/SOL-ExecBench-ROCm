# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Calibrated diagnostic-only IR and hardware prediction."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

from sol_execbench.core.bench.diagnostic_sidecar import DiagnosticSidecarStatus
from sol_execbench.core.bench.performance_model.counter_metrics import (
    counter_memory_bytes,
)
from sol_execbench.core.bench.performance_model.models import (
    CalibrationParameter,
    CompiledCharacterization,
    DiagnosticCalibrationProfile,
    DispatchEvidence,
    PerformancePrediction,
    PredictionComponent,
    PredictionKind,
    SemanticCharacterization,
    WorkloadKind,
)

_IR_PARAMETERS = frozenset(
    {
        "dispatch_floor_ms",
        "valu_flop_per_ms",
        "wmma_flop_per_ms",
        "vram_byte_per_ms",
        "lds_byte_per_ms",
        "reduction_op_per_ms",
    },
)


def validate_calibration_identity(
    profile: DiagnosticCalibrationProfile,
    *,
    gpu_architecture: str,
    gpu_id: str | None = None,
    rocm_version: str | None = None,
    compiler_version: str | None = None,
    clock_mode: str | None = None,
    power_profile: str | None = None,
) -> list[str]:
    """Return explicit reason codes for every calibration identity mismatch."""
    actual = profile.identity
    expected = {
        "gpu_architecture": gpu_architecture,
        "gpu_id": gpu_id,
        "rocm_version": rocm_version,
        "compiler_version": compiler_version,
        "clock_mode": clock_mode,
        "power_profile": power_profile,
    }
    reasons: list[str] = []
    for field, value in expected.items():
        if value is None:
            reasons.append(f"calibration_{field}_unverified")
        elif getattr(actual, field) != value:
            reasons.append(f"calibration_{field}_mismatch")
    return reasons


def predict_ir(
    semantic: SemanticCharacterization,
    calibration: DiagnosticCalibrationProfile,
    *,
    identity_reason_codes: Sequence[str] = (),
) -> PerformancePrediction:
    """Predict logical SOLAR-region runtime without candidate evidence."""
    if identity_reason_codes:
        return _unavailable(PredictionKind.IR, identity_reason_codes)
    parameters, missing = _parameters(calibration, _IR_PARAMETERS)
    if missing:
        return _unavailable(
            PredictionKind.IR,
            [f"missing_calibration_parameter:{name}" for name in missing],
        )
    dispatches = max(len(semantic.fusion_regions), 1)
    components = _semantic_components(semantic, parameters, dispatches)
    predicted, lower, upper = _combine_components(components)
    reasons = list(semantic.reason_codes)
    status = (
        DiagnosticSidecarStatus.PARTIAL
        if semantic.workload_kind is WorkloadKind.UNSUPPORTED or reasons
        else DiagnosticSidecarStatus.AVAILABLE
    )
    return PerformancePrediction(
        kind=PredictionKind.IR,
        status=status,
        predicted_time_ms=predicted,
        lower_ms=lower,
        upper_ms=upper,
        components=components,
        reason_codes=reasons,
        limitations=[
            "Logical dispatches are SOLAR fusion regions.",
            "Prediction is diagnostic-only and is excluded from SOL Score.",
        ],
    )


def predict_hw(
    compiled: Sequence[CompiledCharacterization],
    dispatches: Sequence[DispatchEvidence],
    calibration: DiagnosticCalibrationProfile,
    *,
    identity_reason_codes: Sequence[str] = (),
) -> PerformancePrediction:
    """Predict candidate dispatch runtime without measured/profiler duration."""
    if identity_reason_codes:
        return _unavailable(PredictionKind.HW, identity_reason_codes)
    parameters, missing = _parameters(calibration, _IR_PARAMETERS)
    if missing:
        return _unavailable(
            PredictionKind.HW,
            [f"missing_calibration_parameter:{name}" for name in missing],
        )
    valid = [dispatch for dispatch in dispatches if dispatch.valid]
    if not valid:
        return _unavailable(PredictionKind.HW, ["no_valid_dispatch_evidence"])
    static_by_symbol = {item.kernel_symbol: item for item in compiled}
    modeled = [
        _dispatch_components(
            dispatch,
            static_by_symbol.get(dispatch.kernel_symbol),
            parameters,
        )
        for dispatch in valid
    ]
    components = [component for group in modeled for component in group]
    estimates = [_combine_components(group) for group in modeled]
    predicted, lower, upper = _combine_dispatches(valid, estimates)
    reasons = _hardware_reason_codes(dispatches, components)
    status = (
        DiagnosticSidecarStatus.PARTIAL
        if reasons
        else DiagnosticSidecarStatus.AVAILABLE
    )
    return PerformancePrediction(
        kind=PredictionKind.HW,
        status=status,
        predicted_time_ms=predicted,
        lower_ms=lower,
        upper_ms=upper,
        components=components,
        reason_codes=reasons,
        limitations=[
            "Profiler duration and achieved throughput are not prediction inputs.",
            "Dispatch predictions sum unless timestamps prove overlap.",
        ],
    )


def _parameters(
    calibration: DiagnosticCalibrationProfile,
    required: Iterable[str],
) -> tuple[dict[str, CalibrationParameter], list[str]]:
    parameters = {
        parameter.name: parameter for parameter in calibration.parameters
    }
    missing = sorted(set(required) - parameters.keys())
    return parameters, missing


def _semantic_components(
    semantic: SemanticCharacterization,
    parameters: Mapping[str, CalibrationParameter],
    dispatches: int,
) -> list[PredictionComponent]:
    kind = semantic.workload_kind
    compute_name = (
        "wmma_flop_per_ms"
        if kind is WorkloadKind.MATMUL
        else "valu_flop_per_ms"
    )
    compute = _scaled_component(
        "compute",
        semantic.semantic_flops,
        parameters[compute_name],
    )
    if kind is WorkloadKind.MATMUL:
        efficiency = _wmma_efficiency(semantic, parameters)
        if efficiency is not None:
            compute = _apply_efficiency(compute, efficiency)
    memory = _scaled_component(
        "memory",
        semantic.semantic_bytes,
        _memory_parameter(semantic.semantic_bytes, parameters),
    )
    if kind is WorkloadKind.TRANSPOSE:
        efficiency = parameters.get("transpose_access_efficiency")
        if efficiency is not None:
            memory = _apply_efficiency(memory, efficiency)
    components = [
        _scaled_component(
            "dispatch",
            float(dispatches),
            parameters["dispatch_floor_ms"],
            multiply=True,
        ),
        compute,
        memory,
    ]
    lds_work = _resource_total(semantic.resource_work, "lds")
    reduction_work = _resource_total(semantic.resource_work, "reduction")
    if lds_work:
        components.append(
            _scaled_component(
                "lds",
                lds_work,
                parameters["lds_byte_per_ms"],
            ),
        )
    if reduction_work or kind is WorkloadKind.REDUCTION:
        work = reduction_work or max(semantic.semantic_flops, 1.0)
        components.append(
            _scaled_component(
                "reduction",
                work,
                parameters["reduction_op_per_ms"],
            ),
        )
    return components


def _dispatch_components(
    dispatch: DispatchEvidence,
    compiled: CompiledCharacterization | None,
    parameters: Mapping[str, CalibrationParameter],
) -> list[PredictionComponent]:
    counter = dispatch.counters
    waves = _counter(counter, "SQ_WAVES") or 0.0
    valu_instructions = _counter(
        counter,
        "SQ_INSTS_VALU",
        "SQ_INSTS_VALU_ADD",
    )
    derived_valu_per_wave = _counter(counter, "VALUINSTS")
    if not valu_instructions and derived_valu_per_wave and waves:
        valu_instructions = derived_valu_per_wave * waves
    wmma_instructions = _counter(
        counter,
        "SQ_INSTS_WMMA",
        "SQ_INSTS_MFMA",
        "MFMAINSTS",
    )
    barrier_instructions = 0.0
    lds_instructions = _counter(counter, "SQ_INSTS_LDS")
    if compiled is not None and waves:
        valu_instructions = valu_instructions or (
            _static_group(compiled, "valu") * waves
        )
        wmma_instructions = wmma_instructions or (
            _static_matrix_count(compiled) * waves
        )
        barrier_instructions = _static_group(compiled, "barrier") * waves
        lds_instructions = lds_instructions or (
            _static_group(compiled, "lds") * waves
        )
    dispatch_id = dispatch.dispatch_id
    result = [
        _scaled_component(
            "dispatch",
            1.0,
            parameters["dispatch_floor_ms"],
            multiply=True,
            dispatch_id=dispatch_id,
        ),
    ]
    _append_dynamic_components(
        result,
        dispatch_id=dispatch_id,
        counters=counter,
        valu_instructions=valu_instructions,
        wmma_instructions=wmma_instructions,
        barrier_instructions=barrier_instructions,
        lds_instructions=lds_instructions,
        parameters=parameters,
    )
    return result


def _append_dynamic_components(
    result: list[PredictionComponent],
    *,
    dispatch_id: str,
    counters: Mapping[str, float],
    valu_instructions: float | None,
    wmma_instructions: float | None,
    barrier_instructions: float,
    lds_instructions: float | None,
    parameters: Mapping[str, CalibrationParameter],
) -> None:
    if valu_instructions:
        result.append(
            _scaled_component(
                "compute",
                valu_instructions * 64.0,
                parameters["valu_flop_per_ms"],
                dispatch_id=dispatch_id,
            ),
        )
    if wmma_instructions:
        result.append(
            _scaled_component(
                "wmma",
                wmma_instructions * 8192.0,
                parameters["wmma_flop_per_ms"],
                dispatch_id=dispatch_id,
            ),
        )
    memory_bytes = counter_memory_bytes(counters)
    if memory_bytes:
        result.append(
            _scaled_component(
                "memory",
                memory_bytes,
                parameters["vram_byte_per_ms"],
                dispatch_id=dispatch_id,
            ),
        )
    if lds_instructions:
        lds_component = _scaled_component(
            "lds",
            lds_instructions * 128.0,
            parameters["lds_byte_per_ms"],
            dispatch_id=dispatch_id,
        )
        conflict = _counter(
            counters,
            "LDSBANKCONFLICT",
            "SQC_LDS_BANK_CONFLICT",
            "SQ_LDS_BANK_CONFLICT",
        )
        efficiency = parameters.get("lds_bank_conflict_efficiency")
        if conflict and efficiency is not None:
            lds_component = _apply_efficiency(lds_component, efficiency)
        result.append(lds_component)
    _append_counter_penalties(
        result,
        dispatch_id=dispatch_id,
        counters=counters,
        barrier_instructions=barrier_instructions,
        parameters=parameters,
    )


def _append_counter_penalties(
    result: list[PredictionComponent],
    *,
    dispatch_id: str,
    counters: Mapping[str, float],
    barrier_instructions: float,
    parameters: Mapping[str, CalibrationParameter],
) -> None:
    penalties = (
        (
            "cache",
            _counter(
                counters,
                "GL2C_MISS_SUM",
                "TCC_MISS",
                "L2CACHEMISS",
            ),
            parameters.get("cache_miss_penalty_ms"),
        ),
        (
            "lds_conflict",
            _counter(
                counters,
                "LDSBANKCONFLICT",
                "SQC_LDS_BANK_CONFLICT",
                "SQ_LDS_BANK_CONFLICT",
            ),
            parameters.get("lds_bank_conflict_penalty_ms"),
        ),
    )
    for name, amount, parameter in penalties:
        _append_penalty_component(
            result,
            name=name,
            amount=amount,
            parameter=parameter,
            dispatch_id=dispatch_id,
        )
    _append_penalty_component(
        result,
        name="barrier",
        amount=_counter(counters, "SQ_INSTS_BARRIER", "BARRIERINSTS"),
        fallback_amount=barrier_instructions,
        parameter=parameters.get("barrier_penalty_ms"),
        dispatch_id=dispatch_id,
    )


def _append_penalty_component(
    result: list[PredictionComponent],
    *,
    name: str,
    amount: float | None,
    fallback_amount: float = 0.0,
    parameter: CalibrationParameter | None,
    dispatch_id: str,
) -> None:
    amount = amount or fallback_amount
    if amount and parameter is not None:
        result.append(
            _scaled_component(
                name,
                amount,
                parameter,
                multiply=True,
                dispatch_id=dispatch_id,
            ),
        )


def _scaled_component(
    name: str,
    amount: float,
    parameter: CalibrationParameter,
    *,
    multiply: bool = False,
    dispatch_id: str | None = None,
) -> PredictionComponent:
    lower_parameter, upper_parameter = parameter.confidence_interval
    if multiply:
        estimate = amount * parameter.value
        lower = amount * lower_parameter
        upper = amount * upper_parameter
    else:
        estimate = amount / parameter.value
        lower = amount / upper_parameter
        upper = amount / lower_parameter if lower_parameter else estimate
    return PredictionComponent(
        name=name,
        time_ms=estimate,
        lower_ms=min(lower, estimate),
        upper_ms=max(upper, estimate),
        dispatch_id=dispatch_id,
    )


def _apply_efficiency(
    component: PredictionComponent,
    efficiency: CalibrationParameter,
) -> PredictionComponent:
    lower_efficiency, upper_efficiency = efficiency.confidence_interval
    estimate = component.time_ms / efficiency.value
    lower = component.lower_ms / upper_efficiency
    upper = (
        component.upper_ms / lower_efficiency
        if lower_efficiency
        else component.upper_ms
    )
    return component.model_copy(
        update={
            "time_ms": estimate,
            "lower_ms": min(lower, estimate),
            "upper_ms": max(upper, estimate),
        },
    )


def _memory_parameter(
    working_set_bytes: float,
    parameters: Mapping[str, CalibrationParameter],
) -> CalibrationParameter:
    hierarchy = [
        parameter
        for name in ("l2_byte_per_ms", "l3_byte_per_ms")
        if (parameter := parameters.get(name)) is not None
        and parameter.applicability is not None
        and parameter.applicability[0]
        <= working_set_bytes
        <= parameter.applicability[1]
    ]
    return hierarchy[0] if hierarchy else parameters["vram_byte_per_ms"]


def _wmma_efficiency(
    semantic: SemanticCharacterization,
    parameters: Mapping[str, CalibrationParameter],
) -> CalibrationParameter | None:
    if any("irregular" in reason for reason in semantic.reason_codes):
        return parameters.get("irregular_wmma_efficiency")
    matrix_shape = semantic.shape[-3:]
    if matrix_shape and any(dimension % 16 for dimension in matrix_shape):
        return parameters.get("edge_wmma_efficiency")
    return None


def _combine_components(
    components: Sequence[PredictionComponent],
) -> tuple[float, float, float]:
    dispatch = [item for item in components if item.name == "dispatch"]
    resources = [item for item in components if item.name != "dispatch"]
    dispatch_values = (
        sum(item.time_ms for item in dispatch),
        sum(item.lower_ms for item in dispatch),
        sum(item.upper_ms for item in dispatch),
    )
    if not resources:
        return dispatch_values
    return (
        dispatch_values[0] + max(item.time_ms for item in resources),
        dispatch_values[1] + max(item.lower_ms for item in resources),
        dispatch_values[2] + max(item.upper_ms for item in resources),
    )


def _combine_dispatches(
    dispatches: Sequence[DispatchEvidence],
    estimates: Sequence[tuple[float, float, float]],
) -> tuple[float, float, float]:
    clusters: list[list[int]] = []
    for index, dispatch in enumerate(dispatches):
        overlapping = [
            cluster
            for cluster in clusters
            if any(_overlaps(dispatch, dispatches[other]) for other in cluster)
        ]
        if not overlapping:
            clusters.append([index])
            continue
        merged = [index, *(item for cluster in overlapping for item in cluster)]
        clusters = [
            cluster for cluster in clusters if cluster not in overlapping
        ]
        clusters.append(merged)
    return (
        sum(
            max(estimates[index][0] for index in cluster)
            for cluster in clusters
        ),
        sum(
            max(estimates[index][1] for index in cluster)
            for cluster in clusters
        ),
        sum(
            max(estimates[index][2] for index in cluster)
            for cluster in clusters
        ),
    )


def _overlaps(left: DispatchEvidence, right: DispatchEvidence) -> bool:
    left_start = left.start_timestamp_ns
    left_end = left.end_timestamp_ns
    right_start = right.start_timestamp_ns
    right_end = right.end_timestamp_ns
    if (
        left_start is None
        or left_end is None
        or right_start is None
        or right_end is None
    ):
        return False
    return bool(left_start < right_end and right_start < left_end)


def _hardware_reason_codes(
    dispatches: Sequence[DispatchEvidence],
    components: Sequence[PredictionComponent],
) -> list[str]:
    reasons: list[str] = []
    if any(not dispatch.valid for dispatch in dispatches):
        reasons.append("invalid_dispatch_evidence_excluded")
    component_dispatches = {
        item.dispatch_id for item in components if item.name != "dispatch"
    }
    valid_ids = {item.dispatch_id for item in dispatches if item.valid}
    if component_dispatches != valid_ids:
        reasons.append("missing_dynamic_resource_counters")
    return reasons


def _resource_total(
    resources: Mapping[str, Mapping[str, float]],
    name: str,
) -> float:
    return sum(resources.get(name, {}).values())


def _counter(counters: Mapping[str, float], *names: str) -> float | None:
    return next((counters[name] for name in names if name in counters), None)


def _static_group(compiled: CompiledCharacterization, token: str) -> float:
    return float(
        sum(
            count
            for name, count in compiled.functional_group_counts.items()
            if token in name.lower()
        ),
    )


def _static_matrix_count(compiled: CompiledCharacterization) -> float:
    return float(
        sum(
            count
            for name, count in compiled.functional_subgroup_counts.items()
            if "wmma" in name.lower() or "mfma" in name.lower()
        ),
    )


def _unavailable(
    kind: PredictionKind,
    reasons: Iterable[str],
) -> PerformancePrediction:
    return PerformancePrediction(
        kind=kind,
        status=DiagnosticSidecarStatus.UNAVAILABLE,
        reason_codes=list(reasons),
        limitations=[
            "Prediction is unavailable; no fallback estimate was invented.",
        ],
    )


__all__ = ["predict_hw", "predict_ir", "validate_calibration_identity"]
