# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Paper-aligned Speed-of-Light analysis for PyTorch programs.

The package owns graph extraction, executable extended-einsum conversion,
conversion verification, and hardware-grounded SOL bound derivation.  Dataset
handling, candidate evaluation, timing, baselines, and scoring intentionally
live in :mod:`sol_execbench`.
"""

__version__ = "3.0.0"

# The supported public surface is the atomic pipeline. Stage implementations
# remain importable for in-repository development, but are deliberately not
# advertised as stable entry points because they bypass pipeline attestations.
_LAZY_IMPORTS = {
    "AnalysisFailure": ("solar.api", "AnalysisFailure"),
    "AnalysisRequest": ("solar.api", "AnalysisRequest"),
    "AnalysisResult": ("solar.api", "AnalysisResult"),
    "ArtifactRef": ("solar.api", "ArtifactRef"),
    "SolBound": ("solar.api", "SolBound"),
    "analyze": ("solar.api", "analyze"),
}


def __getattr__(name: str):
    if name not in _LAZY_IMPORTS:
        raise AttributeError(name)
    from importlib import import_module

    module_name, attribute = _LAZY_IMPORTS[name]
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


__all__ = [
    "AnalysisFailure",
    "AnalysisRequest",
    "AnalysisResult",
    "ArtifactRef",
    "SolBound",
    "analyze",
]
