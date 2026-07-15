"""Strict JSON parsing and validation for fusion evidence artifacts."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import cast

from .models import (
    CAPACITY_STATUSES,
    FUSION_VALIDATION_SCHEMA_VERSION,
    FusionCapacityStatus,
    FusionPerformanceStatus,
    PERFORMANCE_STATUSES,
    FusionSignature,
    FusionValidationArtifact,
    FusionValidationCase,
    KernelResourceEvidence,
    PerformanceEvidence,
)
from .performance import performance_from_rounds


def fusion_validation_from_dict(payload: dict[str, object]) -> FusionValidationArtifact:
    """Strictly parse v1 evidence, rejecting unknown fields and forged statuses."""
    _keys(
        payload,
        {
            "schema_version",
            "architecture",
            "gpu_uuid",
            "rocm_version",
            "hipcc_version",
            "clocks_locked",
            "suite_manifest_sha256",
            "benchmark_root_sha256",
            "generated_at",
            "sampling_protocol",
            "cases",
            "profiler_corroboration",
        },
        "fusion validation artifact",
    )
    if payload["schema_version"] != FUSION_VALIDATION_SCHEMA_VERSION:
        raise ValueError("fusion validation artifact has invalid schema_version")
    strings = _non_empty_strings(
        payload,
        ("architecture", "gpu_uuid", "rocm_version", "hipcc_version", "generated_at"),
        "fusion validation artifact",
    )
    suite_manifest_sha256 = _sha256(
        payload["suite_manifest_sha256"], "suite_manifest_sha256"
    )
    benchmark_root_sha256 = _sha256(
        payload["benchmark_root_sha256"], "benchmark_root_sha256"
    )
    clocks_locked = payload["clocks_locked"]
    if not isinstance(clocks_locked, bool):
        raise ValueError("clocks_locked must be boolean")
    _protocol(payload["sampling_protocol"])
    raw_cases = payload["cases"]
    if not isinstance(raw_cases, list):
        raise ValueError("cases must be a list")
    cases = tuple(
        _case(raw, index, strings["architecture"])
        for index, raw in enumerate(raw_cases)
    )
    ids = [case.evidence_id for case in cases]
    signatures = [case.signature.canonical_id for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("cases contains duplicate evidence_id")
    if len(signatures) != len(set(signatures)):
        raise ValueError("cases contains duplicate canonical signature")
    profiler = payload["profiler_corroboration"]
    if profiler is not None and not isinstance(profiler, dict):
        raise ValueError("profiler_corroboration must be an object or null")
    return FusionValidationArtifact(
        architecture=strings["architecture"].lower(),
        gpu_uuid=strings["gpu_uuid"],
        rocm_version=strings["rocm_version"],
        hipcc_version=strings["hipcc_version"],
        clocks_locked=clocks_locked,
        suite_manifest_sha256=suite_manifest_sha256,
        benchmark_root_sha256=benchmark_root_sha256,
        generated_at=strings["generated_at"],
        cases=cases,
        profiler_corroboration=dict(cast(dict[str, object], profiler))
        if profiler is not None
        else None,
    )


def _case(raw: object, index: int, architecture: str) -> FusionValidationCase:
    source = f"cases[{index}]"
    values = _object(raw, source)
    _keys(
        values,
        {
            "evidence_id",
            "workload_uuid",
            "variant_id",
            "signature",
            "signature_sha256",
            "fused",
            "unfused",
            "capacity_status",
            "performance",
            "diagnostics",
        },
        source,
    )
    identifiers = _non_empty_strings(
        values, ("evidence_id", "workload_uuid", "variant_id"), source
    )
    signature = _signature(values["signature"], f"{source}.signature")
    if values["signature_sha256"] != signature.canonical_id:
        raise ValueError(f"{source} signature checksum mismatch")
    fused = _kernel(values["fused"], f"{source}.fused", architecture)
    raw_unfused = values["unfused"]
    if not isinstance(raw_unfused, list) or not raw_unfused:
        raise ValueError(f"{source}.unfused must be a non-empty list")
    unfused = tuple(
        _kernel(kernel, f"{source}.unfused", architecture) for kernel in raw_unfused
    )
    computed_capacity = (
        FusionCapacityStatus.PASSED
        if fused.capacity_passed and all(kernel.capacity_passed for kernel in unfused)
        else FusionCapacityStatus.FAILED
    )
    if (
        values["capacity_status"] not in CAPACITY_STATUSES
        or values["capacity_status"] != computed_capacity
    ):
        raise ValueError(f"{source} capacity_status contradicts resource evidence")
    performance = _performance(values["performance"], f"{source}.performance")
    diagnostics = values["diagnostics"]
    if not isinstance(diagnostics, list) or not all(
        isinstance(item, str) for item in diagnostics
    ):
        raise ValueError(f"{source}.diagnostics must be a string list")
    return FusionValidationCase(
        identifiers["evidence_id"],
        identifiers["workload_uuid"],
        identifiers["variant_id"],
        signature,
        fused,
        unfused,
        computed_capacity,
        performance,
        cast(tuple[str, ...], tuple(diagnostics)),
    )


def _signature(raw: object, source: str) -> FusionSignature:
    values = _object(raw, source)
    _keys(
        values,
        {
            "pattern_id",
            "pattern_version",
            "op_names",
            "dtype",
            "input_shapes",
            "output_shapes",
            "tile_contract",
        },
        source,
    )
    identifiers = _non_empty_strings(values, ("pattern_id", "dtype"), source)
    pattern_version = values["pattern_version"]
    if not isinstance(pattern_version, int) or pattern_version < 1:
        raise ValueError(f"{source}.pattern_version must be positive")
    op_names = values["op_names"]
    if (
        not isinstance(op_names, list)
        or not op_names
        or not all(isinstance(name, str) for name in op_names)
    ):
        raise ValueError(f"{source}.op_names must be a non-empty string list")
    tile_contract = values["tile_contract"]
    if not isinstance(tile_contract, dict):
        raise ValueError(f"{source}.tile_contract must be an object")
    return FusionSignature(
        identifiers["pattern_id"],
        pattern_version,
        cast(tuple[str, ...], tuple(op_names)),
        identifiers["dtype"],
        _shapes(values["input_shapes"], source),
        _shapes(values["output_shapes"], source),
        dict(cast(dict[str, object], tile_contract)),
    )


def _shapes(raw: object, source: str) -> tuple[tuple[int, ...], ...]:
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{source} shapes must be non-empty lists")
    if not all(
        isinstance(shape, list)
        and all(isinstance(dim, int) and dim > 0 for dim in shape)
        for shape in raw
    ):
        raise ValueError(f"{source} contains an invalid shape")
    return tuple(tuple(cast(list[int], shape)) for shape in raw)


def _kernel(raw: object, source: str, architecture: str) -> KernelResourceEvidence:
    values = _object(raw, source)
    _keys(values, set(KernelResourceEvidence.__dataclass_fields__), source)
    strings = _non_empty_strings(
        values,
        ("kernel_name", "source_sha256", "binary_sha256", "architecture"),
        source,
    )
    source_sha256 = _sha256(strings["source_sha256"], f"{source}.source_sha256")
    binary_sha256 = _sha256(strings["binary_sha256"], f"{source}.binary_sha256")
    if strings["architecture"].lower() != architecture.lower():
        raise ValueError(f"{source} architecture mismatch")
    compile_command = values["compile_command"]
    if (
        not isinstance(compile_command, list)
        or not compile_command
        or not all(isinstance(arg, str) for arg in compile_command)
    ):
        raise ValueError(f"{source}.compile_command must be a non-empty string list")
    integer_fields = (
        "vgpr_count",
        "sgpr_count",
        "vgpr_spill_count",
        "sgpr_spill_count",
        "private_segment_bytes",
        "static_lds_bytes",
        "dynamic_lds_bytes",
        "lds_limit_bytes",
        "active_blocks_per_multiprocessor",
    )
    integers = _non_negative_integers(values, integer_fields, source)
    launch_passed, correctness_passed = (
        values["launch_passed"],
        values["correctness_passed"],
    )
    if not isinstance(launch_passed, bool) or not isinstance(correctness_passed, bool):
        raise ValueError(f"{source} launch fields must be boolean")
    return KernelResourceEvidence(
        kernel_name=strings["kernel_name"],
        binary_sha256=binary_sha256,
        source_sha256=source_sha256,
        compile_command=cast(tuple[str, ...], tuple(compile_command)),
        architecture=strings["architecture"].lower(),
        vgpr_count=integers["vgpr_count"],
        sgpr_count=integers["sgpr_count"],
        vgpr_spill_count=integers["vgpr_spill_count"],
        sgpr_spill_count=integers["sgpr_spill_count"],
        private_segment_bytes=integers["private_segment_bytes"],
        static_lds_bytes=integers["static_lds_bytes"],
        dynamic_lds_bytes=integers["dynamic_lds_bytes"],
        lds_limit_bytes=integers["lds_limit_bytes"],
        active_blocks_per_multiprocessor=integers["active_blocks_per_multiprocessor"],
        launch_passed=launch_passed,
        correctness_passed=correctness_passed,
    )


def _performance(raw: object, source: str) -> PerformanceEvidence:
    values = _object(raw, source)
    _keys(values, set(PerformanceEvidence.__dataclass_fields__), source)
    status = values["status"]
    if not isinstance(status, str) or status not in PERFORMANCE_STATUSES:
        raise ValueError(f"{source}.status is invalid")
    performance_status = FusionPerformanceStatus(status)
    protocol = _non_negative_integers(
        values, ("pairs_per_round", "warmups", "process_count"), source
    )
    order = values["order"]
    if not isinstance(order, str):
        raise ValueError(f"{source}.order must be a string")
    if performance_status == FusionPerformanceStatus.NOT_MEASURED:
        if (
            values["fused_round_medians_ms"]
            or values["unfused_round_medians_ms"]
            or values["fused_over_unfused"] is not None
            or values["max_cross_round_spread"] is not None
        ):
            raise ValueError(f"{source} not_measured status contains samples")
        result = PerformanceEvidence(
            performance_status,
            (),
            (),
            None,
            None,
            protocol["pairs_per_round"],
            protocol["warmups"],
            protocol["process_count"],
            order,
        )
    else:
        fused = _round_medians(values["fused_round_medians_ms"], source)
        unfused = _round_medians(values["unfused_round_medians_ms"], source)
        computed = performance_from_rounds(fused, unfused)
        if (
            performance_status != computed.status
            or not math.isclose(
                _float(values["fused_over_unfused"], source),
                computed.fused_over_unfused or 0.0,
                rel_tol=1e-12,
            )
            or not math.isclose(
                _float(values["max_cross_round_spread"], source),
                computed.max_cross_round_spread or 0.0,
                rel_tol=1e-12,
            )
        ):
            raise ValueError(f"{source} status or aggregates contradict samples")
        result = PerformanceEvidence(
            performance_status,
            computed.fused_round_medians_ms,
            computed.unfused_round_medians_ms,
            computed.fused_over_unfused,
            computed.max_cross_round_spread,
            protocol["pairs_per_round"],
            protocol["warmups"],
            protocol["process_count"],
            order,
        )
    if (result.process_count, result.warmups, result.pairs_per_round, result.order) != (
        3,
        10,
        30,
        "alternating_ab_ba",
    ):
        raise ValueError(f"{source} does not use the required sampling protocol")
    return result


def _round_medians(raw: object, source: str) -> tuple[float, ...]:
    if not isinstance(raw, list) or not all(
        isinstance(value, (int, float)) for value in raw
    ):
        raise ValueError(f"{source} round medians must be numeric lists")
    return tuple(float(cast(float, value)) for value in raw)


def _protocol(raw: object) -> None:
    expected = {
        "process_count": 3,
        "warmups": 10,
        "pairs_per_round": 30,
        "order": "alternating_ab_ba",
        "performance_ratio_limit": 1.02,
        "cross_round_spread_limit": 0.05,
    }
    if raw != expected:
        raise ValueError("sampling_protocol does not match the required protocol")


def _object(raw: object, source: str) -> dict[str, object]:
    if not isinstance(raw, dict):
        raise ValueError(f"{source} must be an object")
    return cast(dict[str, object], raw)


def _non_empty_strings(
    raw: Mapping[str, object], keys: tuple[str, ...], source: str
) -> dict[str, str]:
    values = {key: raw[key] for key in keys}
    if not all(isinstance(value, str) and value for value in values.values()):
        raise ValueError(f"{source} identifiers must be non-empty strings")
    return cast(dict[str, str], values)


def _non_negative_integers(
    raw: Mapping[str, object], keys: tuple[str, ...], source: str
) -> dict[str, int]:
    values = {key: raw[key] for key in keys}
    if not all(isinstance(value, int) and value >= 0 for value in values.values()):
        raise ValueError(f"{source} resource values must be non-negative integers")
    return cast(dict[str, int], values)


def _float(raw: object, source: str) -> float:
    if not isinstance(raw, (int, float)):
        raise ValueError(f"{source} aggregate must be numeric")
    return float(raw)


def _sha256(value: object, source: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ValueError(f"{source} must be a lowercase SHA-256")
    return value


def _keys(raw: Mapping[str, object], expected: set[str], source: str) -> None:
    if set(raw) != expected:
        missing, unknown = sorted(expected - set(raw)), sorted(set(raw) - expected)
        raise ValueError(
            f"{source} fields mismatch; missing={missing}, unknown={unknown}"
        )


__all__ = ["fusion_validation_from_dict"]
