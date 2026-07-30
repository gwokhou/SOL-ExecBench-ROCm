"""Runtime evidence sidecar models."""

from __future__ import annotations

from typing import Any, Literal, Protocol

from pydantic import ConfigDict, Field

from sol_execbench.core.data.base_model import (
    BaseModelWithDocstrings,
    StrictArtifactModel,
)

MODEL_CONFIG = ConfigDict(
    extra="forbid",
    frozen=True,
    strict=True,
    use_attribute_docstrings=True,
)

RuntimeFailureCategory = Literal[
    "setup_runtime",
    "dependency",
    "benchmark_correctness",
    "benchmark_performance",
]


class ModelDumpable(Protocol):
    """Minimal Pydantic-compatible serialization protocol."""

    def model_dump(self, *, mode: str) -> dict[str, Any]:
        """Serialize the model using the requested Pydantic mode."""
        ...


class RuntimeFailureEvidence(BaseModelWithDocstrings):
    """Diagnostic failure category recorded outside canonical traces."""

    model_config = MODEL_CONFIG

    category: RuntimeFailureCategory
    """Diagnostic evidence category."""
    status: Literal["recorded"]
    """Fixed status for a recorded diagnostic category."""
    message: str | None = None
    """Optional human-readable diagnostic message."""


class RuntimeGPUTelemetry(StrictArtifactModel):
    """One bounded AMD SMI observation around a diagnostic operation."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        use_attribute_docstrings=True,
        allow_inf_nan=False,
    )

    phase: Literal["pre", "post"]
    gpu_id: str | None = None
    gpu_bdf: str | None = None
    performance_level: str | None = None
    sclk_mhz: float | None = Field(default=None, ge=0)
    mclk_mhz: float | None = Field(default=None, ge=0)
    temperature_c: float | None = Field(default=None, ge=0)
    power_profile: str | None = None
    power_cap_w: float | None = Field(default=None, ge=0)
    power_draw_w: float | None = Field(default=None, ge=0)
    foreign_process_count: int | None = Field(default=None, ge=0)
