# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Focused runtime contracts shared by diagnostic lifecycle handlers."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

from sol_execbench.core.bench.performance_model.lifecycle.enums import (
    DiagnosticEvidencePurpose,
    DiagnosticLifecycleStage,
)
from sol_execbench.core.bench.performance_model.lifecycle.receipts import (
    DiagnosticStageReceipt,
)
from sol_execbench.core.bench.performance_model.lifecycle.run_state import (
    DiagnosticLifecyclePlan,
    DiagnosticRunManifest,
)
from sol_execbench.core.bench.performance_model.lifecycle.shared import (
    DiagnosticLifecycleArtifact,
    DiagnosticLifecycleParent,
)
from sol_execbench.core.integrity import sha256_file
from sol_execbench.core.platform.rdna4_validation import (
    HardwareValidationBinding,
)


@dataclass
class StageRunContext:
    """Resolved run resources plus mutable paths produced during execution.

    Fixed plan values stay owned by ``DiagnosticLifecyclePlan``. Exposing
    them as derived properties avoids a second, independently mutable copy of
    the lifecycle configuration.
    """

    store_root: Path
    plan: DiagnosticLifecyclePlan
    design_manifest_path: Path
    development_corpus_path: Path | None = None
    hardware_validation: HardwareValidationBinding | None = None
    paths: dict[str, Path] = field(default_factory=dict)

    @property
    def collection_run_id(self) -> str:
        """Return the plan-owned collection run identity."""
        return self.plan.collection_run_id

    @property
    def generation(self) -> int:
        """Return the plan-owned collection generation."""
        return self.plan.generation

    @property
    def purpose(self) -> DiagnosticEvidencePurpose:
        """Return the plan-owned evidence purpose."""
        return self.plan.purpose

    @property
    def corpus_root(self) -> Path:
        """Return the reviewed collection root."""
        return Path(self.plan.collection_root)

    @property
    def calibration_profile_path(self) -> Path:
        """Return the reviewed calibration profile path."""
        return Path(self.plan.calibration_profile_path)

    @property
    def calibration_audit_path(self) -> Path:
        """Return the reviewed calibration audit path."""
        return Path(self.plan.calibration_audit_path)

    @property
    def vram_policy_path(self) -> Path | None:
        """Return the optional reviewed VRAM policy path."""
        value = self.plan.vram_policy_path
        return Path(value) if value is not None else None

    @property
    def frozen_inference_profile_path(self) -> Path | None:
        """Return the optional frozen inference profile path."""
        value = self.plan.frozen_inference_profile_path
        return Path(value) if value is not None else None

    @property
    def held_out_corpus_path(self) -> Path:
        """Return the reviewed held-out corpus path."""
        return Path(self.plan.held_out_corpus_path)

    @property
    def output_root(self) -> Path:
        """Return the reviewed output root."""
        return Path(self.plan.output_root)

    @property
    def source_revision(self) -> str:
        """Return the plan-owned source revision."""
        return self.plan.source_revision

    @property
    def model_version(self) -> str:
        """Return the plan-owned model version."""
        return self.plan.model_version

    def output(self, stage: DiagnosticLifecycleStage) -> Path | None:
        """Return the recorded primary output path for one stage."""
        return self.paths.get(stage.value)

    def set_output(self, stage: DiagnosticLifecycleStage, path: Path) -> None:
        """Record the primary output path for one completed stage."""
        self.paths[stage.value] = Path(path)


@dataclass(frozen=True)
class StageCompletion:
    """One completed stage's produced identity and exact output inventory."""

    stage_id: str
    outputs: tuple[DiagnosticLifecycleArtifact, ...]
    output_paths: tuple[Path, ...] = ()


@runtime_checkable
class DiagnosticStageHandler(Protocol):
    """One lifecycle stage adapter: perform, prepare, and re-verify."""

    stage: DiagnosticLifecycleStage

    def run(self, context: StageRunContext) -> StageCompletion:
        """Execute the stage and return its produced identity and outputs."""
        ...

    def prepare(
        self,
        context: StageRunContext,
        run_state: DiagnosticRunManifest,
    ) -> tuple[DiagnosticLifecycleParent, ...]:
        """Return the immutable input identities the stage consumes."""
        ...

    def verify(
        self,
        context: StageRunContext,
        receipt: DiagnosticStageReceipt,
    ) -> bool:
        """Re-verify the completed stage; returns whether it still holds."""
        ...


def artifact_for_path(path: Path) -> DiagnosticLifecycleArtifact:
    """Describe one stage output path for its exact receipt inventory."""
    return DiagnosticLifecycleArtifact(
        relative_path=path.name,
        sha256=sha256_file(path),
        size_bytes=path.stat().st_size,
    )


def verify_artifacts(
    artifacts: Sequence[DiagnosticLifecycleArtifact],
    base: Path,
) -> bool:
    """Re-check that every recorded artifact still verifies under *base*."""
    resolved_base = base.resolve()
    for artifact in artifacts:
        path = (resolved_base / artifact.relative_path).resolve()
        if not path.is_relative_to(resolved_base):
            return False
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != artifact.size_bytes
            or sha256_file(path) != artifact.sha256
        ):
            return False
    return True


__all__ = [
    "DiagnosticStageHandler",
    "StageCompletion",
    "StageRunContext",
    "artifact_for_path",
    "verify_artifacts",
]
