# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
"""CPU-safe low-precision compatibility helpers for migrated datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

LOW_PRECISION_COMPATIBILITY_FORMATS = (
    "nvfp4",
    "mxfp4",
    "float4_e2m1",
    "float4_e2m1fn_x2",
)

LOW_PRECISION_COMPATIBILITY_EVIDENCE_CODE = "phase134_low_precision_cpu_semantics"
CDNA4_VALIDATION_DEFERRED_CODE = "cdna4_low_precision_hardware_validation_deferred"
CDNA3_LOW_PRECISION_SKIP_CODE = "cdna3_low_precision_hardware_unsupported"
CDNA3_GFX_ARCHITECTURES = ("gfx940", "gfx941", "gfx942")

_E2M1_CODE_TO_VALUE = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)


@dataclass(frozen=True)
class LowPrecisionScaleMetadata:
    """Scale metadata for a CPU semantic low-precision payload."""

    scale: float = 1.0
    scale_shape: tuple[int, ...] = ()
    scale_dtype: str = "float32"
    scale_role: str = "global_tensor_scale"

    def to_dict(self) -> dict[str, Any]:
        return {
            "scale": self.scale,
            "scale_shape": list(self.scale_shape),
            "scale_dtype": self.scale_dtype,
            "scale_role": self.scale_role,
        }


@dataclass(frozen=True)
class LowPrecisionCompatibilityEvidence:
    """Evidence boundary for unvalidated low-precision compatibility paths."""

    format_name: str
    evidence_code: str = LOW_PRECISION_COMPATIBILITY_EVIDENCE_CODE
    compatibility_path: str = "cpu_semantic_reference"
    cdna4_validation: str = "deferred_unvalidated"
    blocker_code: str = CDNA4_VALIDATION_DEFERRED_CODE
    hardware_validation: bool = False
    performance_authority: bool = False
    blackwell_equivalence: bool = False
    score_authority: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_name": self.format_name,
            "evidence_code": self.evidence_code,
            "compatibility_path": self.compatibility_path,
            "cdna4_validation": self.cdna4_validation,
            "blocker_code": self.blocker_code,
            "hardware_validation": self.hardware_validation,
            "performance_authority": self.performance_authority,
            "blackwell_equivalence": self.blackwell_equivalence,
            "score_authority": self.score_authority,
            "message": (
                "Low-precision compatibility covers CPU semantic packing and reference behavior only; "
                "CDNA4 hardware validation and performance authority remain deferred."
            ),
        }


@dataclass(frozen=True)
class PackedLowPrecisionTensor:
    """Packed two-values-per-byte E2M1 payload with original shape metadata."""

    format_name: str
    packed: torch.Tensor
    original_shape: tuple[int, ...]
    scale_metadata: LowPrecisionScaleMetadata
    evidence: LowPrecisionCompatibilityEvidence

    def unpack_codes(self) -> torch.Tensor:
        return unpack_e2m1_codes(self.packed, self.original_shape)

    def dequantize(self) -> torch.Tensor:
        codes = self.unpack_codes()
        return dequantize_e2m1_codes(codes, scale=self.scale_metadata.scale)

    def metadata_dict(self) -> dict[str, Any]:
        return {
            "format_name": self.format_name,
            "packed_shape": list(self.packed.shape),
            "original_shape": list(self.original_shape),
            "scale_metadata": self.scale_metadata.to_dict(),
            "evidence": self.evidence.to_dict(),
        }


def normalize_low_precision_format(format_name: str) -> str:
    normalized = format_name.lower().replace("-", "_")
    aliases = {
        "nv_fp4": "nvfp4",
        "mx_fp4": "mxfp4",
        "fp4": "float4_e2m1",
        "e2m1": "float4_e2m1",
        "e2m1_x2": "float4_e2m1fn_x2",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in LOW_PRECISION_COMPATIBILITY_FORMATS:
        raise ValueError(f"Unsupported low-precision compatibility format: {format_name}")
    return normalized


def low_precision_unvalidated_evidence(format_name: str = "nvfp4") -> LowPrecisionCompatibilityEvidence:
    return LowPrecisionCompatibilityEvidence(format_name=normalize_low_precision_format(format_name))


def definition_uses_cdna4_low_precision(definition: dict[str, Any]) -> bool:
    """Return whether a problem definition uses NVFP4/MXFP4-style FP4 formats."""

    tokens: set[str] = set()
    tokens.add(str(definition.get("name", "")))
    tokens.add(str(definition.get("reference", "")))
    for section_name in ("inputs", "outputs"):
        section = definition.get(section_name, {})
        if not isinstance(section, dict):
            continue
        for value in section.values():
            if not isinstance(value, dict):
                continue
            tokens.add(str(value.get("dtype", "")))
            tokens.add(str(value.get("description", "")))
    return any(_contains_low_precision_token(token) for token in tokens)


def should_skip_cdna4_low_precision_on_arch(
    definition: dict[str, Any], gpu_architecture: str | None
) -> bool:
    """Return whether a CDNA3 run should skip CDNA4-only low-precision problems."""

    arch = (gpu_architecture or "").lower()
    return arch.startswith(CDNA3_GFX_ARCHITECTURES) and definition_uses_cdna4_low_precision(
        definition
    )


def cdna4_low_precision_skip_reason(gpu_architecture: str | None) -> str:
    arch = gpu_architecture or "unknown"
    return (
        f"{CDNA3_LOW_PRECISION_SKIP_CODE}: {arch} does not provide default "
        "NVFP4/MXFP4 Quant validation support; run on CDNA4-class hardware for "
        "hardware validation."
    )


def quantize_e2m1_codes(values: torch.Tensor, *, scale: float = 1.0) -> torch.Tensor:
    """Quantize float values to unsigned 4-bit E2M1 codes."""

    if scale <= 0:
        raise ValueError("scale must be positive")
    scaled = values.detach().to(dtype=torch.float32, device="cpu") / scale
    result = torch.zeros_like(scaled, dtype=torch.uint8)

    result[(scaled >= 0.0) & (scaled <= 0.25)] = 0
    result[(scaled > 0.25) & (scaled < 0.75)] = 1
    result[(scaled >= 0.75) & (scaled <= 1.25)] = 2
    result[(scaled > 1.25) & (scaled < 1.75)] = 3
    result[(scaled >= 1.75) & (scaled <= 2.5)] = 4
    result[(scaled > 2.5) & (scaled < 3.5)] = 5
    result[(scaled >= 3.5) & (scaled <= 5.0)] = 6
    result[scaled > 5.0] = 7

    result[(scaled >= -0.25) & (scaled < 0.0)] = 8
    result[(scaled < -0.25) & (scaled > -0.75)] = 9
    result[(scaled <= -0.75) & (scaled >= -1.25)] = 10
    result[(scaled < -1.25) & (scaled > -1.75)] = 11
    result[(scaled <= -1.75) & (scaled >= -2.5)] = 12
    result[(scaled < -2.5) & (scaled > -3.5)] = 13
    result[(scaled <= -3.5) & (scaled >= -5.0)] = 14
    result[scaled < -5.0] = 15
    return result


def dequantize_e2m1_codes(codes: torch.Tensor, *, scale: float = 1.0) -> torch.Tensor:
    """Convert unsigned 4-bit E2M1 codes back to float32 values."""

    if scale <= 0:
        raise ValueError("scale must be positive")
    uint_codes = codes.detach().to(dtype=torch.uint8, device="cpu")
    _validate_e2m1_codes(uint_codes)
    codebook = _E2M1_CODE_TO_VALUE.to(device=uint_codes.device)
    return codebook[uint_codes.long()] * scale


def pack_e2m1_codes(codes: torch.Tensor) -> torch.Tensor:
    """Pack E2M1 codes into uint8 with the even element in the low nibble."""

    uint_codes = codes.detach().to(dtype=torch.uint8, device="cpu")
    _validate_e2m1_codes(uint_codes)
    if uint_codes.ndim == 0:
        uint_codes = uint_codes.reshape(1)
    if uint_codes.shape[-1] % 2:
        pad_shape = (*uint_codes.shape[:-1], 1)
        uint_codes = torch.cat([uint_codes, torch.zeros(pad_shape, dtype=torch.uint8)], dim=-1)
    low = uint_codes[..., 0::2]
    high = uint_codes[..., 1::2] << 4
    return low | high


def unpack_e2m1_codes(packed: torch.Tensor, original_shape: tuple[int, ...] | list[int]) -> torch.Tensor:
    """Unpack uint8 E2M1 pairs and restore the original uncompressed shape."""

    shape = tuple(int(dim) for dim in original_shape)
    for dim in shape:
        if dim < 0:
            raise ValueError("original_shape dimensions must be non-negative")

    uint_packed = packed.detach().to(dtype=torch.uint8, device="cpu")
    expected_packed_shape = _packed_shape(shape)
    if tuple(uint_packed.shape) != expected_packed_shape:
        raise ValueError("packed tensor does not contain enough values for original_shape")
    if not shape:
        return (uint_packed & 0x0F).reshape(())
    unpacked_shape = (*shape[:-1], expected_packed_shape[-1] * 2)
    unpacked = torch.empty(unpacked_shape, dtype=torch.uint8)
    unpacked[..., 0::2] = uint_packed & 0x0F
    unpacked[..., 1::2] = uint_packed >> 4
    return unpacked[..., : shape[-1]]


def pack_low_precision_tensor(
    values: torch.Tensor,
    *,
    format_name: str = "nvfp4",
    scale: float = 1.0,
    scale_shape: tuple[int, ...] = (),
    scale_dtype: str = "float32",
    scale_role: str = "global_tensor_scale",
) -> PackedLowPrecisionTensor:
    """Quantize and pack a tensor with explicit unvalidated-CDNA4 evidence."""

    normalized = normalize_low_precision_format(format_name)
    codes = quantize_e2m1_codes(values, scale=scale)
    scale_metadata = LowPrecisionScaleMetadata(
        scale=scale,
        scale_shape=scale_shape,
        scale_dtype=scale_dtype,
        scale_role=scale_role,
    )
    return PackedLowPrecisionTensor(
        format_name=normalized,
        packed=pack_e2m1_codes(codes),
        original_shape=tuple(values.shape),
        scale_metadata=scale_metadata,
        evidence=low_precision_unvalidated_evidence(normalized),
    )


def _contains_low_precision_token(value: str) -> bool:
    normalized = value.lower().replace("-", "_")
    tokens = (
        *LOW_PRECISION_COMPATIBILITY_FORMATS,
        "float4",
        "e2m1",
        "float8_e4m3fn",
        "scaled_gemm",
        "_scaled_mm",
    )
    return any(token in normalized for token in tokens)


def _packed_shape(shape: torch.Size | tuple[int, ...]) -> tuple[int, ...]:
    if not shape:
        return (1,)
    leading = tuple(int(dim) for dim in shape[:-1])
    last_dim = int(shape[-1])
    return leading + ((last_dim + 1) // 2,)


def _validate_e2m1_codes(codes: torch.Tensor) -> None:
    if bool((codes > 15).any().item()):
        raise ValueError("E2M1 codes must be in the range [0, 15]")
