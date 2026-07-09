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

"""Data layer compatibility facade for SOL ExecBench."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "AxisConst": ".definition",
    "AxisExpr": ".definition",
    "AxisSpec": ".definition",
    "AxisVar": ".definition",
    "TensorSpec": ".definition",
    "Definition": ".definition",
    "SOL_EXECBENCH_CONTRACT_SCHEMA_VERSION": ".contract",
    "SOL_EXECBENCH_CONTRACT_VERSION": ".contract",
    "EvaluatorContract": ".contract",
    "build_evaluator_contract": ".contract",
    "SourceFile": ".solution",
    "BuildSpec": ".solution",
    "CompileOptions": ".solution",
    "SupportedBindings": ".solution",
    "SupportedHardware": ".solution",
    "SupportedLanguages": ".solution",
    "Solution": ".solution",
    "ToleranceSpec": ".workload",
    "CustomInput": ".workload",
    "RandomInput": ".workload",
    "ScalarInput": ".workload",
    "SafetensorsInput": ".workload",
    "InputSpec": ".workload",
    "Workload": ".workload",
    "Correctness": ".trace",
    "Performance": ".trace",
    "Environment": ".trace",
    "Evaluation": ".trace",
    "EvaluationStatus": ".trace",
    "Trace": ".trace",
    "save_json_file": ".json_utils",
    "load_json_file": ".json_utils",
    "save_jsonl_file": ".json_utils",
    "load_jsonl_file": ".json_utils",
    "append_jsonl_file": ".json_utils",
}

__all__ = [
    # Definition types
    "AxisConst",
    "AxisExpr",
    "AxisSpec",
    "AxisVar",
    "TensorSpec",
    "Definition",
    # Contract types
    "SOL_EXECBENCH_CONTRACT_SCHEMA_VERSION",
    "SOL_EXECBENCH_CONTRACT_VERSION",
    "EvaluatorContract",
    "build_evaluator_contract",
    # Solution types
    "SourceFile",
    "BuildSpec",
    "CompileOptions",
    "SupportedBindings",
    "SupportedHardware",
    "SupportedLanguages",
    "Solution",
    # Workload types
    "ToleranceSpec",
    "CustomInput",
    "RandomInput",
    "ScalarInput",
    "SafetensorsInput",
    "InputSpec",
    "Workload",
    # Trace types
    "Correctness",
    "Performance",
    "Environment",
    "Evaluation",
    "EvaluationStatus",
    "Trace",
    # JSON functions
    "save_json_file",
    "load_json_file",
    "save_jsonl_file",
    "load_jsonl_file",
    "append_jsonl_file",
]


def __getattr__(name: str) -> Any:
    """Load compatibility re-exports on first access."""
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Return stable names for interactive discovery and star imports."""
    return sorted({*globals(), *__all__})
