# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Stable typed failures shared by SOLAR stage boundaries."""

from __future__ import annotations

from typing import ClassVar


class SolarError(RuntimeError, ValueError):
    """Base class for failures carrying a stable public reason code."""

    reason_code: ClassVar[str] = "solar_failed"


class SourceInputBindingError(SolarError):
    """Source arguments cannot be bound exactly to graph inputs."""

    reason_code = "source_input_binding_failed"


class ReferenceOutputBindingError(SolarError):
    """Reference outputs cannot be bound exactly to graph outputs."""

    reason_code = "reference_output_binding_failed"


class UnsupportedOperationError(SolarError):
    """An operation has no exact implementation in the selected IR."""

    reason_code = "exact_operation_unsupported"


class NativeOperationUnsupportedError(SolarError):
    """An Extended native target has no reviewed capability entry."""

    reason_code = "native_operation_unsupported"


class NativeAttributeUnsupportedError(SolarError):
    """An Extended native call contains an unreviewed public parameter."""

    reason_code = "native_attribute_unsupported"


class DynamicShapeUnboundedError(SolarError):
    """A data-dependent dimension has no consistent finite bounds."""

    reason_code = "dynamic_shape_unbounded"


class GradientVerificationError(SolarError):
    """Reference and IR first-order gradients are not equivalent."""

    reason_code = "gradient_verification_failed"


class StrictConversionError(SolarError):
    """An operator graph cannot be converted under the strict contract."""

    reason_code = "strict_conversion_failed"


class ConversionVerificationError(SolarError):
    """A conversion could not be proven equivalent."""

    reason_code = "conversion_not_proven"


class IRReplayError(ConversionVerificationError):
    """An IR graph cannot replay exactly."""

    reason_code = "exact_replay_failed"


class NumericalEquivalenceError(ConversionVerificationError):
    """Reference and IR outputs violate the numerical policy."""

    reason_code = "numerical_equivalence_failed"


class ToolchainUnavailableError(SolarError):
    """A required external proof tool did not produce valid evidence."""

    reason_code = "toolchain_unavailable"


__all__ = [
    "ConversionVerificationError",
    "DynamicShapeUnboundedError",
    "GradientVerificationError",
    "IRReplayError",
    "NativeAttributeUnsupportedError",
    "NativeOperationUnsupportedError",
    "NumericalEquivalenceError",
    "ReferenceOutputBindingError",
    "SolarError",
    "SourceInputBindingError",
    "StrictConversionError",
    "ToolchainUnavailableError",
    "UnsupportedOperationError",
]
