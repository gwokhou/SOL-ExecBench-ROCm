# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Core public convenience exports."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "AxisConst": ".data",
    "AxisSpec": ".data",
    "AxisVar": ".data",
    "BuildSpec": ".data",
    "CompileOptions": ".data",
    "Correctness": ".data",
    "CustomInput": ".data",
    "Definition": ".data",
    "Environment": ".data",
    "Evaluation": ".data",
    "EvaluationStatus": ".data",
    "InputSpec": ".data",
    "Performance": ".data",
    "RandomInput": ".data",
    "SafetensorsInput": ".data",
    "ScalarInput": ".data",
    "Solution": ".data",
    "SourceFile": ".data",
    "SupportedBindings": ".data",
    "SupportedHardware": ".data",
    "SupportedLanguages": ".data",
    "TensorSpec": ".data",
    "ToleranceSpec": ".data",
    "Trace": ".data",
    "Workload": ".data",
    "ENVIRONMENT_SNAPSHOT_SCHEMA_VERSION": ".platform.environment",
    "EnvironmentCheckResult": ".platform.environment",
    "EnvironmentDiagnostics": ".platform.environment",
    "EnvironmentEvidenceStatus": ".platform.environment",
    "EnvironmentSnapshot": ".platform.environment",
    "GpuEnvironmentSummary": ".platform.environment",
    "ProbeCompletedProcess": ".platform.environment",
    "PytorchRocmSummary": ".platform.environment",
    "RocmEnvironmentSummary": ".platform.environment",
    "ToolProbeResult": ".platform.environment",
    "build_environment_diagnostics": ".platform.environment",
    "collect_environment_snapshot": ".platform.environment",
    "collect_pytorch_rocm_summary": ".platform.environment",
    "probe_tool": ".platform.environment",
    "TOOLCHAIN_ROUTING_SCHEMA_VERSION": ".platform.toolchain",
    "ToolLifecycle": ".platform.toolchain",
    "ToolchainArtifactType": ".platform.toolchain",
    "ToolchainCapability": ".platform.toolchain",
    "ToolchainEvidenceLevel": ".platform.toolchain",
    "ToolchainProbeResult": ".platform.toolchain",
    "ToolchainRoutingDecision": ".platform.toolchain",
    "ToolchainRoutingReport": ".platform.toolchain",
    "ToolchainRoutingRequest": ".platform.toolchain",
    "ToolchainStatus": ".platform.toolchain",
    "build_toolchain_routing_report": ".platform.toolchain",
    "default_toolchain_registry": ".platform.toolchain",
    "probe_toolchain_tool": ".platform.toolchain",
    "BenchmarkConfig": ".bench.config",
}

__all__ = [
    # Data models
    "AxisConst",
    "AxisSpec",
    "AxisVar",
    "TensorSpec",
    "Definition",
    "SourceFile",
    "BuildSpec",
    "CompileOptions",
    "SupportedBindings",
    "SupportedHardware",
    "SupportedLanguages",
    "Solution",
    "ToleranceSpec",
    "RandomInput",
    "ScalarInput",
    "SafetensorsInput",
    "CustomInput",
    "InputSpec",
    "Workload",
    "Correctness",
    "Performance",
    "Environment",
    "Evaluation",
    "EvaluationStatus",
    "Trace",
    # Optional environment evidence
    "ENVIRONMENT_SNAPSHOT_SCHEMA_VERSION",
    "EnvironmentCheckResult",
    "EnvironmentDiagnostics",
    "EnvironmentEvidenceStatus",
    "EnvironmentSnapshot",
    "GpuEnvironmentSummary",
    "ProbeCompletedProcess",
    "PytorchRocmSummary",
    "RocmEnvironmentSummary",
    "ToolProbeResult",
    "build_environment_diagnostics",
    "collect_environment_snapshot",
    "collect_pytorch_rocm_summary",
    "probe_tool",
    # Optional toolchain routing evidence
    "TOOLCHAIN_ROUTING_SCHEMA_VERSION",
    "ToolLifecycle",
    "ToolchainArtifactType",
    "ToolchainCapability",
    "ToolchainEvidenceLevel",
    "ToolchainProbeResult",
    "ToolchainRoutingDecision",
    "ToolchainRoutingReport",
    "ToolchainRoutingRequest",
    "ToolchainStatus",
    "build_toolchain_routing_report",
    "default_toolchain_registry",
    "probe_toolchain_tool",
    # Bench config
    "BenchmarkConfig",
]


def __getattr__(name: str) -> Any:
    """Load convenience exports on first access."""
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Return stable names for interactive discovery and star imports."""
    return sorted({*globals(), *__all__})
