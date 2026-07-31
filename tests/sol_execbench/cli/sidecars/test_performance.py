from __future__ import annotations

from typing import Literal

from sol_execbench.cli.sidecars.performance import _gpu_identity
from sol_execbench.core.evidence.runtime_evidence.models import (
    RuntimeGPUTelemetry,
)


def _snapshot(
    phase: Literal["pre", "post"],
    *,
    gpu_id: str | None = "gpu-0",
    gpu_bdf: str | None = "0000:03:00.0",
) -> RuntimeGPUTelemetry:
    return RuntimeGPUTelemetry.model_validate(
        {
            "phase": phase,
            "gpu_id": gpu_id,
            "gpu_bdf": gpu_bdf,
        },
    )


def test_gpu_identity_binds_matching_pre_and_post_snapshots() -> None:
    identity = _gpu_identity((_snapshot("pre"), _snapshot("post")))

    assert identity == ("gpu-0", "0000:03:00.0", [])


def test_gpu_identity_rejects_snapshot_drift() -> None:
    identity = _gpu_identity(
        (
            _snapshot("pre"),
            _snapshot("post", gpu_id="gpu-1"),
        ),
    )

    assert identity == (None, None, ["gpu_id_snapshot_invalid"])
