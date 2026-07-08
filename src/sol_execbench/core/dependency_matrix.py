"""PyTorch ROCm dependency policy helpers for declared Docker Targets."""

from __future__ import annotations

from sol_execbench.core.dependency_matrix_classification import classify_dependency_preflight
from sol_execbench.core.dependency_matrix_cli import main
from sol_execbench.core.dependency_matrix_collectors import collect_pytorch_dependency_observation
from sol_execbench.core.dependency_matrix_models import (
    DependencyPreflightResult,
    PytorchDependencyObservation,
    PytorchDependencyPolicy,
)
from sol_execbench.core.dependency_matrix_policy import (
    dependency_policy_evidence_for_target,
    load_docker_target_dependency_policy,
)

__all__ = [
    "DependencyPreflightResult",
    "PytorchDependencyObservation",
    "PytorchDependencyPolicy",
    "classify_dependency_preflight",
    "collect_pytorch_dependency_observation",
    "dependency_policy_evidence_for_target",
    "load_docker_target_dependency_policy",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
