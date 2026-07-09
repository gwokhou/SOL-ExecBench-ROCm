"""Runtime evidence sidecar models."""

from __future__ import annotations

from typing import Any, Literal, Protocol

from pydantic import ConfigDict

from sol_execbench.core.data.base_model import BaseModelWithDocstrings


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
    def model_dump(self, *, mode: str) -> dict[str, Any]: ...


class RuntimeFailureEvidence(BaseModelWithDocstrings):
    """Diagnostic failure category recorded outside canonical traces."""

    model_config = MODEL_CONFIG

    category: RuntimeFailureCategory
    """Diagnostic evidence category."""
    status: str
    """Category-specific status value."""
    message: str | None = None
    """Optional human-readable diagnostic message."""
