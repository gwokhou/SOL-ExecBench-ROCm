"""Fusion validation schema models."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .io import sha256_payload

FUSION_VALIDATION_SCHEMA_VERSION = "sol_execbench.fusion_validation.v1"


class FusionCapacityStatus(str, Enum):
    """Capacity-validation outcomes admitted by fusion evidence."""

    PASSED = "passed"
    FAILED = "failed"


class FusionPerformanceStatus(str, Enum):
    """Performance-validation outcomes admitted by fusion evidence."""

    PASSED = "passed"
    FAILED = "failed"
    UNSTABLE = "unstable"
    NOT_MEASURED = "not_measured"


# Kept as value sets for strict parsing and backwards-compatible public imports.
CAPACITY_STATUSES = frozenset(status.value for status in FusionCapacityStatus)
PERFORMANCE_STATUSES = frozenset(status.value for status in FusionPerformanceStatus)


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
        object.__setattr__(
            self, "dtype", aliases.get(self.dtype.lower(), self.dtype.lower())
        )

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
    status: FusionPerformanceStatus
    fused_round_medians_ms: tuple[float, ...]
    unfused_round_medians_ms: tuple[float, ...]
    fused_over_unfused: float | None
    max_cross_round_spread: float | None
    pairs_per_round: int = 30
    warmups: int = 10
    process_count: int = 3
    order: str = "alternating_ab_ba"

    def __post_init__(self) -> None:
        """Coerce serialized values and reject invalid in-memory evidence."""

        object.__setattr__(self, "status", FusionPerformanceStatus(self.status))

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status.value,
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
    capacity_status: FusionCapacityStatus
    performance: PerformanceEvidence
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Coerce serialized values and reject invalid in-memory evidence."""

        object.__setattr__(
            self,
            "capacity_status",
            FusionCapacityStatus(self.capacity_status),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "evidence_id": self.evidence_id,
            "workload_uuid": self.workload_uuid,
            "variant_id": self.variant_id,
            "signature": self.signature.to_dict(),
            "signature_sha256": self.signature.canonical_id,
            "fused": self.fused.to_dict(),
            "unfused": [kernel.to_dict() for kernel in self.unfused],
            "capacity_status": self.capacity_status.value,
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


__all__ = [
    "CAPACITY_STATUSES",
    "FUSION_VALIDATION_SCHEMA_VERSION",
    "PERFORMANCE_STATUSES",
    "FusionCapacityStatus",
    "FusionPerformanceStatus",
    "FusionSignature",
    "FusionValidationArtifact",
    "FusionValidationCase",
    "KernelResourceEvidence",
    "PerformanceEvidence",
]
