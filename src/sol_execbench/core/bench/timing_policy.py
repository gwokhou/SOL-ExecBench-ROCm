# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Timing source classification and policy selection for ROCm benchmarks.

This module is intentionally pure. It defines the timing semantics that later
profiler-backed execution can consume without launching profilers or touching
benchmark traces.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from typing import Any

from sol_execbench.core.data.solution import SupportedLanguages


class TimingSourceType(str, Enum):
    """Internal source categories that determine timing interpretation."""

    PYTORCH = "pytorch"
    TRITON = "triton"
    HIP_NATIVE = "hip_native"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class TimingBackend(str, Enum):
    """Timer backend selected for a timing source."""

    ROCPROFV3 = "rocprofv3"
    PYTORCH_PROFILER = "pytorch_profiler"
    DEVICE_EVENTS = "device_events"
    UNSUPPORTED = "unsupported"


class TimingActivityDomain(str, Enum):
    """What layer of work the selected timing backend measures."""

    KERNEL_ACTIVITY = "kernel_activity"
    HIP_RUNTIME_API = "hip_runtime_api"
    PYTORCH_OPERATOR_ATTRIBUTION = "pytorch_operator_attribution"
    FALLBACK_EVENT_TIMING = "fallback_event_timing"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class TimingPolicy:
    """Selected timing policy and user-facing interpretation."""

    source_type: TimingSourceType
    backend: TimingBackend
    activity_domain: TimingActivityDomain
    aggregation_rule: str
    interpretation: str
    fallback_applied: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable policy payload."""
        return {
            "source_type": self.source_type.value,
            "backend": self.backend.value,
            "activity_domain": self.activity_domain.value,
            "aggregation_rule": self.aggregation_rule,
            "interpretation": self.interpretation,
            "fallback_applied": self.fallback_applied,
            "reason": self.reason,
        }


_PYTHON_LANGUAGES = {
    SupportedLanguages.PYTORCH,
    SupportedLanguages.TRITON,
}

_NATIVE_LANGUAGES = {
    SupportedLanguages.HIP_CPP,
    SupportedLanguages.HIPBLAS,
    SupportedLanguages.MIOPEN,
    SupportedLanguages.CK,
    SupportedLanguages.ROCWMMA,
}


def _coerce_language(language: SupportedLanguages | str) -> SupportedLanguages | None:
    if isinstance(language, SupportedLanguages):
        return language
    try:
        return SupportedLanguages(language)
    except ValueError:
        return None


def classify_timing_source(
    languages: Iterable[SupportedLanguages | str],
) -> TimingSourceType:
    """Classify public solution languages into an internal timing source type."""
    parsed = {_coerce_language(language) for language in languages}
    parsed.discard(None)
    if not parsed:
        return TimingSourceType.UNKNOWN

    has_python = bool(parsed & _PYTHON_LANGUAGES)
    has_native = bool(parsed & _NATIVE_LANGUAGES)
    if has_python and has_native:
        return TimingSourceType.MIXED
    if SupportedLanguages.TRITON in parsed:
        return TimingSourceType.TRITON
    if SupportedLanguages.PYTORCH in parsed:
        return TimingSourceType.PYTORCH
    if has_native:
        return TimingSourceType.HIP_NATIVE
    return TimingSourceType.UNKNOWN


def _event_fallback_policy(
    source_type: TimingSourceType,
    *,
    reason: str,
) -> TimingPolicy:
    return TimingPolicy(
        source_type=source_type,
        backend=TimingBackend.DEVICE_EVENTS,
        activity_domain=TimingActivityDomain.FALLBACK_EVENT_TIMING,
        aggregation_rule="median of HIP-backed PyTorch device event elapsed times",
        interpretation=(
            "fallback event timing around the benchmark callable; this is not "
            "profiler-backed kernel activity timing"
        ),
        fallback_applied=True,
        reason=reason,
    )


def select_timing_policy(
    source_type: TimingSourceType,
    *,
    profiler_available: bool = True,
) -> TimingPolicy:
    """Return the timing policy for *source_type*.

    ``profiler_available=False`` models the labeled fallback path. It does not
    probe local tools.
    """
    if not profiler_available:
        return _event_fallback_policy(
            source_type,
            reason="profiler-backed timing is unavailable",
        )

    if source_type == TimingSourceType.HIP_NATIVE:
        return TimingPolicy(
            source_type=source_type,
            backend=TimingBackend.ROCPROFV3,
            activity_domain=TimingActivityDomain.KERNEL_ACTIVITY,
            aggregation_rule=(
                "aggregate ROCm kernel activity rows launched by the measured "
                "solution call"
            ),
            interpretation=(
                "kernel activity duration for native HIP or ROCm library work "
                "inside the benchmark timing region"
            ),
            fallback_applied=False,
            reason="HIP native source is best measured from ROCm kernel activity",
        )

    if source_type == TimingSourceType.TRITON:
        return TimingPolicy(
            source_type=source_type,
            backend=TimingBackend.ROCPROFV3,
            activity_domain=TimingActivityDomain.KERNEL_ACTIVITY,
            aggregation_rule=(
                "aggregate post-warmup ROCm kernel activity rows for generated "
                "Triton kernels launched by the measured solution call"
            ),
            interpretation=(
                "kernel activity duration for Triton-generated kernels, "
                "excluding compile and autotune warmup unless explicitly labeled"
            ),
            fallback_applied=False,
            reason="Triton source requires generated-kernel activity semantics",
        )

    if source_type == TimingSourceType.PYTORCH:
        return TimingPolicy(
            source_type=source_type,
            backend=TimingBackend.PYTORCH_PROFILER,
            activity_domain=TimingActivityDomain.PYTORCH_OPERATOR_ATTRIBUTION,
            aggregation_rule=(
                "attribute PyTorch operator regions and cross-check associated "
                "device activity"
            ),
            interpretation=(
                "PyTorch operator attribution for ROCm device work; operator "
                "regions can dispatch multiple HIP or library kernels"
            ),
            fallback_applied=False,
            reason="PyTorch source needs operator attribution in addition to device work",
        )

    if source_type == TimingSourceType.MIXED:
        return _event_fallback_policy(
            source_type,
            reason="mixed source timing requires runtime evidence before profiler selection",
        )

    return TimingPolicy(
        source_type=source_type,
        backend=TimingBackend.UNSUPPORTED,
        activity_domain=TimingActivityDomain.UNSUPPORTED,
        aggregation_rule="unsupported until source type is classified",
        interpretation="no accurate timer can be selected for an unknown source",
        fallback_applied=True,
        reason="source type is unknown",
    )


def timing_policy_for_languages(
    languages: Iterable[SupportedLanguages | str],
    *,
    profiler_available: bool = True,
) -> TimingPolicy:
    """Classify *languages* and return the selected timing policy."""
    return select_timing_policy(
        classify_timing_source(languages),
        profiler_available=profiler_available,
    )


def timing_policy_table(
    *,
    profiler_available: bool = True,
) -> tuple[TimingPolicy, ...]:
    """Return inspectable policies for every internal source type."""
    return tuple(
        select_timing_policy(source_type, profiler_available=profiler_available)
        for source_type in TimingSourceType
    )
