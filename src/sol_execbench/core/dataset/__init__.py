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
from .evidence_refs import (
    build_derived_evidence_refs,
    relative_ref,
    safe_sidecar_stem,
)
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
from .ready_subset import ReadySubset, build_ready_subset, write_ready_subset
from .sharding import (
    DatasetShardMergeResult,
    DatasetShardPlan,
    DatasetShardWorkload,
    merge_dataset_shard_traces,
    plan_dataset_shards,
)
from .parity_gap import (
    ParityGapReport,
    build_parity_gap_report,
    render_parity_gap_markdown,
    write_parity_gap_reports,
)
from .paper_denominator import (
    PaperDenominatorReport,
    build_paper_denominator_report,
    render_paper_denominator_markdown,
    write_paper_denominator_reports,
)

__all__ = [
    "DEFAULT_CATEGORIES",
    "DatasetLayout",
    "DatasetManifest",
    "DatasetManifestSource",
    "DatasetInventory",
    "DatasetReadiness",
    "DatasetShardMergeResult",
    "DatasetShardPlan",
    "DatasetShardWorkload",
    "InventoryDenominators",
    "LayoutCategory",
    "LayoutDiagnostic",
    "ProblemInventoryRecord",
    "ParityGapReport",
    "PaperDenominatorReport",
    "ReadySubset",
    "WorkloadInventoryRecord",
    "WorkloadReadinessRecord",
    "build_derived_evidence_refs",
    "build_dataset_manifest",
    "build_dataset_inventory",
    "classify_rocm_readiness",
    "build_ready_subset",
    "merge_dataset_shard_traces",
    "plan_dataset_shards",
    "build_parity_gap_report",
    "build_paper_denominator_report",
    "inspect_dataset_layout",
    "render_parity_gap_markdown",
    "render_paper_denominator_markdown",
    "relative_ref",
    "safe_sidecar_stem",
    "validate_categories",
    "write_dataset_manifest",
    "write_dataset_inventory",
    "write_dataset_readiness",
    "write_ready_subset",
    "write_parity_gap_reports",
    "write_paper_denominator_reports",
]
