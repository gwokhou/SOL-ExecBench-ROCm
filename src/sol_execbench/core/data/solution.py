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

"""Strong-typed data definitions for solution implementations."""

import hashlib
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from pydantic import ConfigDict, Field, PrivateAttr, field_validator, model_validator


from sol_execbench.core.data.solution_instance import Solution
from sol_execbench.core.data.solution_models import (
    NATIVE_ROCM_LANGUAGES,
    BuildSpec,
    CompileOptions,
    SourceFile,
    SupportedBindings,
    SupportedHardware,
    SupportedLanguages,
)

__all__ = [
    "NATIVE_ROCM_LANGUAGES",
    "BuildSpec",
    "CompileOptions",
    "Solution",
    "SourceFile",
    "SupportedBindings",
    "SupportedHardware",
    "SupportedLanguages",
]
