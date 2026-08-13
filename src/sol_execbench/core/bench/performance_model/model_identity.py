# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Narrow, content-addressed identity for performance inference behavior."""

from __future__ import annotations

import importlib
from importlib.resources import as_file, files
from pathlib import Path

from sol_execbench.core.bench.performance_model.models import (
    DiagnosticModelIdentity,
)
from sol_execbench.core.integrity import (
    sha256_file,
    stable_json_checksum,
)

_POLICY_MODULES = (
    "sol_execbench.core.bench.performance_model.prediction",
    "sol_execbench.core.bench.performance_model.attribution",
    "sol_execbench.core.bench.performance_model.counter_metrics",
    "sol_execbench.core.bench.performance_model.inference",
)
_DEFAULT_COUNTER_RESOURCE = "gfx1200_v3.yaml"


def build_diagnostic_model_identity(
    model_version: str,
    *,
    counter_resource: str = _DEFAULT_COUNTER_RESOURCE,
) -> DiagnosticModelIdentity:
    """Hash the declared inference owners without binding orchestration code.

    ``counter_resource`` binds the per-architecture counter manifest into the
    identity digest. gfx1200 callers keep the default; gfx942 callers pass
    ``gfx942_v1.yaml`` so ``counter_semantics_sha256`` stays per-architecture.
    """
    policy_files = {
        module_name: sha256_file(_module_path(module_name))
        for module_name in _POLICY_MODULES
    }
    resource = files("sol_execbench.data.rocprofv3_counters").joinpath(
        counter_resource,
    )
    with as_file(resource) as resource_path:
        counter_hash = sha256_file(resource_path)
    bundle_hash = stable_json_checksum(
        {
            "model_version": model_version,
            "policy_files": policy_files,
            "counter_semantics_sha256": counter_hash,
        },
    )
    return DiagnosticModelIdentity(
        model_version=model_version,
        policy_files=policy_files,
        counter_semantics_sha256=counter_hash,
        policy_bundle_sha256=bundle_hash,
    )


def _module_path(module_name: str) -> Path:
    module = importlib.import_module(module_name)
    source = getattr(module, "__file__", None)
    if source is None:
        raise RuntimeError(f"model policy module has no source: {module_name}")
    return Path(source)


__all__ = [
    "DiagnosticModelIdentity",
    "build_diagnostic_model_identity",
]
