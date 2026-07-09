"""Runtime evidence sidecars and aggregate compatibility reports."""

from __future__ import annotations

from sol_execbench.core.platform.dependency_matrix import (
    collect_pytorch_dependency_observation,
)
from sol_execbench.core.evidence.runtime_evidence.builders import (
    build_aggregate_report,
    build_runtime_matrix_entry,
)
from sol_execbench.core.evidence.runtime_evidence.cli import (
    collect_target as _collect_target_impl,
    main as _main,
)
from sol_execbench.core.evidence.runtime_evidence.collectors import (
    build_dependency_observation as _build_dependency_observation,
    build_host_evidence,
    collect_gpu_evidence,
    collect_visible_device_environment,
)
from sol_execbench.core.evidence.runtime_evidence.io import (
    load_matrix_entry,
    write_aggregate_report,
    write_json_payload,
    write_matrix_entry,
)
from sol_execbench.core.evidence.runtime_evidence.models import (
    RuntimeFailureCategory,
    RuntimeFailureEvidence,
)


def build_dependency_observation(**kwargs):
    """Build dependency observations from injected values or local packages."""
    return _build_dependency_observation(
        **kwargs,
        collect_observation=collect_pytorch_dependency_observation,
    )


def _collect_target(args):
    return _collect_target_impl(
        args, build_dependency_observation=build_dependency_observation
    )


def main(argv: list[str] | None = None) -> int:
    """Emit runtime evidence sidecars and aggregate reports."""
    return _main(argv, build_dependency_observation=build_dependency_observation)


__all__ = [
    "RuntimeFailureCategory",
    "RuntimeFailureEvidence",
    "build_aggregate_report",
    "build_dependency_observation",
    "build_host_evidence",
    "build_runtime_matrix_entry",
    "collect_gpu_evidence",
    "collect_pytorch_dependency_observation",
    "collect_visible_device_environment",
    "load_matrix_entry",
    "main",
    "write_aggregate_report",
    "write_json_payload",
    "write_matrix_entry",
]


if __name__ == "__main__":
    raise SystemExit(main())
