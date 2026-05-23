# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Dataset acquisition and layout contract helpers."""

from .categories import DEFAULT_CATEGORIES, validate_categories
from .layout import (
    DatasetLayout,
    LayoutCategory,
    LayoutDiagnostic,
    inspect_dataset_layout,
)
from .manifest import (
    DatasetManifest,
    DatasetManifestSource,
    build_dataset_manifest,
    write_dataset_manifest,
)
from .inventory import (
    DatasetInventory,
    InventoryDenominators,
    ProblemInventoryRecord,
    WorkloadInventoryRecord,
    build_dataset_inventory,
    write_dataset_inventory,
)
from .readiness import (
    DatasetReadiness,
    WorkloadReadinessRecord,
    classify_rocm_readiness,
    write_dataset_readiness,
)

__all__ = [
    "DEFAULT_CATEGORIES",
    "DatasetLayout",
    "DatasetManifest",
    "DatasetManifestSource",
    "DatasetInventory",
    "DatasetReadiness",
    "InventoryDenominators",
    "LayoutCategory",
    "LayoutDiagnostic",
    "ProblemInventoryRecord",
    "WorkloadInventoryRecord",
    "WorkloadReadinessRecord",
    "build_dataset_manifest",
    "build_dataset_inventory",
    "classify_rocm_readiness",
    "inspect_dataset_layout",
    "validate_categories",
    "write_dataset_manifest",
    "write_dataset_inventory",
    "write_dataset_readiness",
]
