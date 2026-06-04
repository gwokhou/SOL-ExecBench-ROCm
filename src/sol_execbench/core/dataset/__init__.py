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
from .low_precision import (
    CDNA4_VALIDATION_DEFERRED_CODE,
    LOW_PRECISION_COMPATIBILITY_EVIDENCE_CODE,
    LOW_PRECISION_COMPATIBILITY_FORMATS,
    LowPrecisionCompatibilityEvidence,
    LowPrecisionScaleMetadata,
    PackedLowPrecisionTensor,
    dequantize_e2m1_codes,
    low_precision_unvalidated_evidence,
    normalize_low_precision_format,
    pack_e2m1_codes,
    pack_low_precision_tensor,
    quantize_e2m1_codes,
    unpack_e2m1_codes,
)
from .manifest import (
    DatasetManifest,
    DatasetManifestSource,
    build_dataset_manifest,
    write_dataset_manifest,
)
from .migration import (
    DatasetMigrationManifest,
    migrate_flashinfer_trace,
    migrate_sol_execbench,
    write_migration_manifest,
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
    DatasetReadinessClaimBoundary,
    ReadinessBlockerReport,
    ReadinessClass,
    WorkloadReadinessRecord,
    classify_rocm_readiness,
    write_dataset_readiness,
)
from .ready_subset import (
    ReadySubset,
    ReadySubsetDenominator,
    ReadySubsetExclusionReason,
    build_ready_subset,
    write_ready_subset,
)
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
    "DatasetMigrationManifest",
    "DatasetInventory",
    "DatasetReadiness",
    "DatasetReadinessClaimBoundary",
    "DatasetShardMergeResult",
    "DatasetShardPlan",
    "DatasetShardWorkload",
    "CDNA4_VALIDATION_DEFERRED_CODE",
    "InventoryDenominators",
    "LayoutCategory",
    "LayoutDiagnostic",
    "LOW_PRECISION_COMPATIBILITY_EVIDENCE_CODE",
    "LOW_PRECISION_COMPATIBILITY_FORMATS",
    "LowPrecisionCompatibilityEvidence",
    "LowPrecisionScaleMetadata",
    "ProblemInventoryRecord",
    "ParityGapReport",
    "PaperDenominatorReport",
    "PackedLowPrecisionTensor",
    "ReadySubset",
    "ReadySubsetDenominator",
    "ReadySubsetExclusionReason",
    "ReadinessBlockerReport",
    "ReadinessClass",
    "WorkloadInventoryRecord",
    "WorkloadReadinessRecord",
    "build_derived_evidence_refs",
    "build_dataset_manifest",
    "build_dataset_inventory",
    "classify_rocm_readiness",
    "build_ready_subset",
    "dequantize_e2m1_codes",
    "low_precision_unvalidated_evidence",
    "migrate_flashinfer_trace",
    "migrate_sol_execbench",
    "merge_dataset_shard_traces",
    "normalize_low_precision_format",
    "plan_dataset_shards",
    "pack_e2m1_codes",
    "pack_low_precision_tensor",
    "quantize_e2m1_codes",
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
    "write_migration_manifest",
    "write_dataset_readiness",
    "write_ready_subset",
    "unpack_e2m1_codes",
    "write_parity_gap_reports",
    "write_paper_denominator_reports",
]
