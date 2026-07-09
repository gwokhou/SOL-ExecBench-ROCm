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

"""Public SOL ExecBench compatibility facade."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "AxisConst": ".core",
    "AxisSpec": ".core",
    "AxisVar": ".core",
    "BenchmarkConfig": ".core",
    "BuildSpec": ".core",
    "CompileOptions": ".core",
    "Correctness": ".core",
    "CustomInput": ".core",
    "Definition": ".core",
    "Environment": ".core",
    "Evaluation": ".core",
    "EvaluationStatus": ".core",
    "InputSpec": ".core",
    "Performance": ".core",
    "RandomInput": ".core",
    "SafetensorsInput": ".core",
    "ScalarInput": ".core",
    "Solution": ".core",
    "SourceFile": ".core",
    "SupportedBindings": ".core",
    "SupportedHardware": ".core",
    "SupportedLanguages": ".core",
    "TensorSpec": ".core",
    "ToleranceSpec": ".core",
    "Trace": ".core",
    "Workload": ".core",
    "get_clock_preset": ".core",
    "append_jsonl_file": ".core.data",
    "load_json_file": ".core.data",
    "load_jsonl_file": ".core.data",
    "save_json_file": ".core.data",
    "save_jsonl_file": ".core.data",
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
    # Bench config
    "BenchmarkConfig",
    "get_clock_preset",
    # JSON utilities
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
