# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Authoritative, shape-exact validation evidence for AMD fusion groups.

Unlike the static-kernel sidecar this artifact is allowed to prove capacity: it
binds code-object resources, HIP occupancy and an actual correctness launch to
one architecture, device and exact fusion signature.  Performance is recorded
separately and never changes the capacity decision.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from sol_execbench.core.bench.static_kernel.amdgpu_metadata import (
    extract_amdgpu_kernel_metadata,
)


FUSION_VALIDATION_SCHEMA_VERSION = "sol_execbench.fusion_validation.v1"
CAPACITY_STATUSES = frozenset({"passed", "failed"})
PERFORMANCE_STATUSES = frozenset({"passed", "failed", "unstable", "not_measured"})


def canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    """Return the canonical representation used by evidence checksums."""
    return (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()


def sha256_payload(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def kernel_resource_from_code_object(
    *,
    code_object: bytes,
    source: bytes,
    kernel_name: str,
    compile_command: tuple[str, ...],
    architecture: str,
    dynamic_lds_bytes: int,
    lds_limit_bytes: int,
    active_blocks_per_multiprocessor: int,
    launch_passed: bool,
    correctness_passed: bool,
) -> KernelResourceEvidence:
    """Build authority evidence after an exact, unique metadata match."""
    matches = [
        item
        for item in extract_amdgpu_kernel_metadata(
            code_object, target_architecture=architecture
        )
        if item.name == kernel_name or item.symbol == kernel_name
    ]
    if len(matches) != 1:
        raise ValueError(
            f"kernel {kernel_name!r} must uniquely match gfx code-object metadata"
        )
    metadata = matches[0]
    if (
        metadata.vgpr_count is None
        or metadata.sgpr_count is None
        or metadata.private_segment_bytes is None
        or metadata.group_segment_bytes is None
    ):
        raise ValueError(f"kernel {kernel_name!r} has incomplete resource metadata")
    return KernelResourceEvidence(
        kernel_name=metadata.name,
        binary_sha256=hashlib.sha256(code_object).hexdigest(),
        source_sha256=hashlib.sha256(source).hexdigest(),
        compile_command=compile_command,
        architecture=metadata.architecture,
        vgpr_count=metadata.vgpr_count,
        sgpr_count=metadata.sgpr_count,
        vgpr_spill_count=metadata.vgpr_spill_count,
        sgpr_spill_count=metadata.sgpr_spill_count,
        private_segment_bytes=metadata.private_segment_bytes,
        static_lds_bytes=metadata.group_segment_bytes,
        dynamic_lds_bytes=dynamic_lds_bytes,
        lds_limit_bytes=lds_limit_bytes,
        active_blocks_per_multiprocessor=active_blocks_per_multiprocessor,
        launch_passed=launch_passed,
        correctness_passed=correctness_passed,
    )


@dataclass(frozen=True)
class FusionSignature:
    """Canonical identity of one graph pattern at one exact workload shape."""

    pattern_id: str
    pattern_version: int
    op_names: tuple[str, ...]
    dtype: str
    input_shapes: tuple[tuple[int, ...], ...]
    output_shapes: tuple[tuple[int, ...], ...]
    tile_contract: dict[str, object]

    def __post_init__(self) -> None:
        aliases = {
            "float32": "fp32",
            "torch.float32": "fp32",
            "float16": "fp16",
            "torch.float16": "fp16",
            "bfloat16": "bf16",
            "torch.bfloat16": "bf16",
        }
        canonical = aliases.get(self.dtype.lower(), self.dtype.lower())
        object.__setattr__(self, "dtype", canonical)

    def to_dict(self) -> dict[str, object]:
        return {
            "pattern_id": self.pattern_id,
            "pattern_version": self.pattern_version,
            "op_names": list(self.op_names),
            "dtype": self.dtype,
            "input_shapes": [list(shape) for shape in self.input_shapes],
            "output_shapes": [list(shape) for shape in self.output_shapes],
            "tile_contract": dict(sorted(self.tile_contract.items())),
        }

    @property
    def canonical_id(self) -> str:
        return sha256_payload(self.to_dict())


@dataclass(frozen=True)
class KernelResourceEvidence:
    kernel_name: str
    binary_sha256: str
    source_sha256: str
    compile_command: tuple[str, ...]
    architecture: str
    vgpr_count: int
    sgpr_count: int
    vgpr_spill_count: int
    sgpr_spill_count: int
    private_segment_bytes: int
    static_lds_bytes: int
    dynamic_lds_bytes: int
    lds_limit_bytes: int
    active_blocks_per_multiprocessor: int
    launch_passed: bool
    correctness_passed: bool

    @property
    def capacity_passed(self) -> bool:
        return (
            self.vgpr_spill_count == 0
            and self.sgpr_spill_count == 0
            and self.private_segment_bytes == 0
            and self.static_lds_bytes + self.dynamic_lds_bytes <= self.lds_limit_bytes
            and self.active_blocks_per_multiprocessor >= 1
            and self.launch_passed
            and self.correctness_passed
        )

    def to_dict(self) -> dict[str, object]:
        return {
            key: (list(value) if isinstance(value, tuple) else value)
            for key, value in self.__dict__.items()
        }


@dataclass(frozen=True)
class PerformanceEvidence:
    status: str
    fused_round_medians_ms: tuple[float, ...]
    unfused_round_medians_ms: tuple[float, ...]
    fused_over_unfused: float | None
    max_cross_round_spread: float | None
    pairs_per_round: int = 30
    warmups: int = 10
    process_count: int = 3
    order: str = "alternating_ab_ba"

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "fused_round_medians_ms": list(self.fused_round_medians_ms),
            "unfused_round_medians_ms": list(self.unfused_round_medians_ms),
            "fused_over_unfused": self.fused_over_unfused,
            "max_cross_round_spread": self.max_cross_round_spread,
            "pairs_per_round": self.pairs_per_round,
            "warmups": self.warmups,
            "process_count": self.process_count,
            "order": self.order,
        }


@dataclass(frozen=True)
class FusionValidationCase:
    evidence_id: str
    workload_uuid: str
    variant_id: str
    signature: FusionSignature
    fused: KernelResourceEvidence
    unfused: tuple[KernelResourceEvidence, ...]
    capacity_status: str
    performance: PerformanceEvidence
    diagnostics: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "evidence_id": self.evidence_id,
            "workload_uuid": self.workload_uuid,
            "variant_id": self.variant_id,
            "signature": self.signature.to_dict(),
            "signature_sha256": self.signature.canonical_id,
            "fused": self.fused.to_dict(),
            "unfused": [kernel.to_dict() for kernel in self.unfused],
            "capacity_status": self.capacity_status,
            "performance": self.performance.to_dict(),
            "diagnostics": list(self.diagnostics),
        }


@dataclass(frozen=True)
class FusionValidationArtifact:
    architecture: str
    gpu_uuid: str
    rocm_version: str
    hipcc_version: str
    clocks_locked: bool
    suite_manifest_sha256: str
    benchmark_root_sha256: str
    generated_at: str
    cases: tuple[FusionValidationCase, ...]
    profiler_corroboration: dict[str, object] | None = None
    schema_version: str = FUSION_VALIDATION_SCHEMA_VERSION

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "architecture": self.architecture,
            "gpu_uuid": self.gpu_uuid,
            "rocm_version": self.rocm_version,
            "hipcc_version": self.hipcc_version,
            "clocks_locked": self.clocks_locked,
            "suite_manifest_sha256": self.suite_manifest_sha256,
            "benchmark_root_sha256": self.benchmark_root_sha256,
            "generated_at": self.generated_at,
            "sampling_protocol": {
                "process_count": 3,
                "warmups": 10,
                "pairs_per_round": 30,
                "order": "alternating_ab_ba",
                "performance_ratio_limit": 1.02,
                "cross_round_spread_limit": 0.05,
            },
            "cases": [case.to_dict() for case in self.cases],
            "profiler_corroboration": self.profiler_corroboration,
        }

    def matching_case(
        self, signature: FusionSignature, *, workload_uuid: str | None = None
    ) -> FusionValidationCase | None:
        matches = [
            case
            for case in self.cases
            if case.signature == signature
            and (workload_uuid is None or case.workload_uuid == workload_uuid)
        ]
        return matches[0] if len(matches) == 1 else None


def performance_from_rounds(
    fused_round_medians_ms: tuple[float, ...],
    unfused_round_medians_ms: tuple[float, ...],
) -> PerformanceEvidence:
    """Apply the fixed 3-round stability and 2% paired performance policy."""
    if len(fused_round_medians_ms) != 3 or len(unfused_round_medians_ms) != 3:
        raise ValueError("performance evidence requires exactly three process rounds")
    values = fused_round_medians_ms + unfused_round_medians_ms
    if not all(math.isfinite(value) and value > 0 for value in values):
        raise ValueError("performance medians must be finite and positive")

    def median(items: tuple[float, ...]) -> float:
        return sorted(items)[1]

    def spread(items: tuple[float, ...]) -> float:
        center = median(items)
        return (max(items) - min(items)) / center

    ratio = median(fused_round_medians_ms) / median(unfused_round_medians_ms)
    max_spread = max(spread(fused_round_medians_ms), spread(unfused_round_medians_ms))
    status = (
        "unstable" if max_spread > 0.05 else ("passed" if ratio <= 1.02 else "failed")
    )
    return PerformanceEvidence(
        status=status,
        fused_round_medians_ms=fused_round_medians_ms,
        unfused_round_medians_ms=unfused_round_medians_ms,
        fused_over_unfused=ratio,
        max_cross_round_spread=max_spread,
    )


def fusion_validation_from_dict(payload: dict[str, Any]) -> FusionValidationArtifact:
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
    for key in (
        "architecture",
        "gpu_uuid",
        "rocm_version",
        "hipcc_version",
        "generated_at",
    ):
        if not isinstance(payload[key], str) or not payload[key]:
            raise ValueError(f"{key} must be a non-empty string")
    for key in ("suite_manifest_sha256", "benchmark_root_sha256"):
        _hash(payload[key], key)
    if not isinstance(payload["clocks_locked"], bool):
        raise ValueError("clocks_locked must be boolean")
    _protocol(payload["sampling_protocol"])
    if not isinstance(payload["cases"], list):
        raise ValueError("cases must be a list")
    cases = tuple(
        _case(item, index, payload["architecture"])
        for index, item in enumerate(payload["cases"])
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
        architecture=payload["architecture"].lower(),
        gpu_uuid=payload["gpu_uuid"],
        rocm_version=payload["rocm_version"],
        hipcc_version=payload["hipcc_version"],
        clocks_locked=payload["clocks_locked"],
        suite_manifest_sha256=payload["suite_manifest_sha256"],
        benchmark_root_sha256=payload["benchmark_root_sha256"],
        generated_at=payload["generated_at"],
        cases=cases,
        profiler_corroboration=dict(profiler) if profiler is not None else None,
    )


def _case(raw: Any, index: int, architecture: str) -> FusionValidationCase:
    if not isinstance(raw, dict):
        raise ValueError(f"cases[{index}] must be an object")
    _keys(
        raw,
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
        f"cases[{index}]",
    )
    if not all(
        isinstance(raw[key], str) and raw[key]
        for key in ("evidence_id", "workload_uuid", "variant_id")
    ):
        raise ValueError(f"cases[{index}] identifiers must be non-empty strings")
    signature = _signature(raw["signature"], f"cases[{index}].signature")
    if raw["signature_sha256"] != signature.canonical_id:
        raise ValueError(f"cases[{index}] signature checksum mismatch")
    fused = _kernel(raw["fused"], f"cases[{index}].fused", architecture)
    if not isinstance(raw["unfused"], list) or not raw["unfused"]:
        raise ValueError(f"cases[{index}].unfused must be a non-empty list")
    unfused = tuple(
        _kernel(item, f"cases[{index}].unfused", architecture)
        for item in raw["unfused"]
    )
    computed_capacity = (
        "passed"
        if fused.capacity_passed and all(kernel.capacity_passed for kernel in unfused)
        else "failed"
    )
    if (
        raw["capacity_status"] not in CAPACITY_STATUSES
        or raw["capacity_status"] != computed_capacity
    ):
        raise ValueError(
            f"cases[{index}] capacity_status contradicts resource evidence"
        )
    performance = _performance(raw["performance"], f"cases[{index}].performance")
    if not isinstance(raw["diagnostics"], list) or not all(
        isinstance(item, str) for item in raw["diagnostics"]
    ):
        raise ValueError(f"cases[{index}].diagnostics must be a string list")
    return FusionValidationCase(
        raw["evidence_id"],
        raw["workload_uuid"],
        raw["variant_id"],
        signature,
        fused,
        unfused,
        computed_capacity,
        performance,
        tuple(raw["diagnostics"]),
    )


def _signature(raw: Any, source: str) -> FusionSignature:
    if not isinstance(raw, dict):
        raise ValueError(f"{source} must be an object")
    _keys(
        raw,
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
    if not isinstance(raw["pattern_id"], str) or not isinstance(raw["dtype"], str):
        raise ValueError(f"{source} identifiers must be strings")
    if not isinstance(raw["pattern_version"], int) or raw["pattern_version"] < 1:
        raise ValueError(f"{source}.pattern_version must be positive")
    if (
        not isinstance(raw["op_names"], list)
        or not raw["op_names"]
        or not all(isinstance(x, str) for x in raw["op_names"])
    ):
        raise ValueError(f"{source}.op_names must be a non-empty string list")
    if not isinstance(raw["tile_contract"], dict):
        raise ValueError(f"{source}.tile_contract must be an object")
    return FusionSignature(
        raw["pattern_id"],
        raw["pattern_version"],
        tuple(raw["op_names"]),
        raw["dtype"],
        _shapes(raw["input_shapes"], source),
        _shapes(raw["output_shapes"], source),
        dict(raw["tile_contract"]),
    )


def _shapes(raw: Any, source: str) -> tuple[tuple[int, ...], ...]:
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{source} shapes must be non-empty lists")
    if not all(
        isinstance(shape, list)
        and shape
        and all(isinstance(dim, int) and dim > 0 for dim in shape)
        for shape in raw
    ):
        raise ValueError(f"{source} contains an invalid shape")
    return tuple(tuple(shape) for shape in raw)


def _kernel(raw: Any, source: str, architecture: str) -> KernelResourceEvidence:
    if not isinstance(raw, dict):
        raise ValueError(f"{source} must be an object")
    fields = set(KernelResourceEvidence.__dataclass_fields__)
    _keys(raw, fields, source)
    for key in ("kernel_name", "source_sha256", "binary_sha256", "architecture"):
        if not isinstance(raw[key], str) or not raw[key]:
            raise ValueError(f"{source}.{key} must be a non-empty string")
    _hash(raw["source_sha256"], f"{source}.source_sha256")
    _hash(raw["binary_sha256"], f"{source}.binary_sha256")
    if raw["architecture"].lower() != architecture.lower():
        raise ValueError(f"{source} architecture mismatch")
    if (
        not isinstance(raw["compile_command"], list)
        or not raw["compile_command"]
        or not all(isinstance(x, str) for x in raw["compile_command"])
    ):
        raise ValueError(f"{source}.compile_command must be a non-empty string list")
    ints = (
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
    if not all(isinstance(raw[key], int) and raw[key] >= 0 for key in ints):
        raise ValueError(f"{source} resource values must be non-negative integers")
    if not all(
        isinstance(raw[key], bool) for key in ("launch_passed", "correctness_passed")
    ):
        raise ValueError(f"{source} launch fields must be boolean")
    return KernelResourceEvidence(
        kernel_name=raw["kernel_name"],
        binary_sha256=raw["binary_sha256"],
        source_sha256=raw["source_sha256"],
        compile_command=tuple(raw["compile_command"]),
        architecture=raw["architecture"].lower(),
        vgpr_count=raw["vgpr_count"],
        sgpr_count=raw["sgpr_count"],
        vgpr_spill_count=raw["vgpr_spill_count"],
        sgpr_spill_count=raw["sgpr_spill_count"],
        private_segment_bytes=raw["private_segment_bytes"],
        static_lds_bytes=raw["static_lds_bytes"],
        dynamic_lds_bytes=raw["dynamic_lds_bytes"],
        lds_limit_bytes=raw["lds_limit_bytes"],
        active_blocks_per_multiprocessor=raw["active_blocks_per_multiprocessor"],
        launch_passed=raw["launch_passed"],
        correctness_passed=raw["correctness_passed"],
    )


def _performance(raw: Any, source: str) -> PerformanceEvidence:
    if not isinstance(raw, dict):
        raise ValueError(f"{source} must be an object")
    _keys(raw, set(PerformanceEvidence.__dataclass_fields__), source)
    status = raw["status"]
    if status not in PERFORMANCE_STATUSES:
        raise ValueError(f"{source}.status is invalid")
    if status == "not_measured":
        if (
            raw["fused_round_medians_ms"]
            or raw["unfused_round_medians_ms"]
            or raw["fused_over_unfused"] is not None
            or raw["max_cross_round_spread"] is not None
        ):
            raise ValueError(f"{source} not_measured status contains samples")
        result = PerformanceEvidence(
            status,
            (),
            (),
            None,
            None,
            raw["pairs_per_round"],
            raw["warmups"],
            raw["process_count"],
            raw["order"],
        )
    else:
        computed = performance_from_rounds(
            tuple(raw["fused_round_medians_ms"]), tuple(raw["unfused_round_medians_ms"])
        )
        assert computed.fused_over_unfused is not None
        assert computed.max_cross_round_spread is not None
        if (
            status != computed.status
            or not math.isclose(
                raw["fused_over_unfused"], computed.fused_over_unfused, rel_tol=1e-12
            )
            or not math.isclose(
                raw["max_cross_round_spread"],
                computed.max_cross_round_spread,
                rel_tol=1e-12,
            )
        ):
            raise ValueError(f"{source} status or aggregates contradict samples")
        result = PerformanceEvidence(
            status,
            computed.fused_round_medians_ms,
            computed.unfused_round_medians_ms,
            computed.fused_over_unfused,
            computed.max_cross_round_spread,
            raw["pairs_per_round"],
            raw["warmups"],
            raw["process_count"],
            raw["order"],
        )
    if (result.process_count, result.warmups, result.pairs_per_round, result.order) != (
        3,
        10,
        30,
        "alternating_ab_ba",
    ):
        raise ValueError(f"{source} does not use the required sampling protocol")
    return result


def _protocol(raw: Any) -> None:
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


def _hash(value: Any, source: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ValueError(f"{source} must be a lowercase SHA-256")


def _keys(raw: Mapping[str, Any], expected: set[str], source: str) -> None:
    if set(raw) != expected:
        missing, unknown = sorted(expected - set(raw)), sorted(set(raw) - expected)
        raise ValueError(
            f"{source} fields mismatch; missing={missing}, unknown={unknown}"
        )
