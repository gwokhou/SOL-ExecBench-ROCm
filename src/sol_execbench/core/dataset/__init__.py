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

"""Dataset acquisition and layout contract helpers.

This package exposes dataset subsystem symbols without eagerly importing every
submodule at package import time. Internal code should import from the focused
submodule that owns the symbol.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "DEFAULT_CATEGORIES": ".categories",
    "DatasetLayout": ".layout",
    "DatasetManifest": ".manifest",
    "DatasetManifestSource": ".manifest",
    "DatasetMigrationManifest": ".migration",
    "DatasetInventory": ".inventory",
    "DatasetReadiness": ".readiness",
    "DatasetReadinessClaimBoundary": ".readiness",
    "DatasetShardMergeResult": ".sharding",
    "DatasetShardPlan": ".sharding",
    "DatasetShardWorkload": ".sharding",
    "CDNA4_VALIDATION_DEFERRED_CODE": ".low_precision",
    "CDNA3_LOW_PRECISION_SKIP_CODE": ".low_precision",
    "InventoryDenominators": ".inventory",
    "LayoutCategory": ".layout",
    "LayoutDiagnostic": ".layout",
    "LOW_PRECISION_COMPATIBILITY_EVIDENCE_CODE": ".low_precision",
    "LOW_PRECISION_COMPATIBILITY_FORMATS": ".low_precision",
    "LONG_TAIL_EXCLUSION_REASON": ".long_tail_exclusions",
    "LONG_TAIL_EXCLUSION_SCHEMA_VERSION": ".long_tail_exclusions",
    "LONG_TAIL_EXCLUSION_STATUS": ".long_tail_exclusions",
    "LongTailExclusionConfig": ".long_tail_exclusions",
    "LongTailExclusionEntry": ".long_tail_exclusions",
    "LongTailExclusionSidecar": ".long_tail_exclusions",
    "LowPrecisionCompatibilityEvidence": ".low_precision",
    "LowPrecisionScaleMetadata": ".low_precision",
    "ProblemInventoryRecord": ".inventory",
    "ParityGapReport": ".parity_gap",
    "PaperDenominatorReport": ".paper_denominator",
    "ProfilerTimingCoverageReport": ".profiler_timing_coverage",
    "PackedLowPrecisionTensor": ".low_precision",
    "ReadySubset": ".ready_subset",
    "ReadySubsetDenominator": ".ready_subset",
    "ReadySubsetExclusionReason": ".ready_subset",
    "ReadinessBlockerReport": ".readiness",
    "ReadinessClass": ".readiness",
    "WorkloadInventoryRecord": ".inventory",
    "WorkloadReadinessRecord": ".readiness",
    "build_derived_evidence_refs": ".evidence_refs",
    "build_dataset_manifest": ".manifest",
    "build_dataset_inventory": ".inventory",
    "classify_rocm_readiness": ".readiness",
    "build_ready_subset": ".ready_subset",
    "cdna4_low_precision_skip_reason": ".low_precision",
    "dequantize_e2m1_codes": ".low_precision",
    "definition_uses_cdna4_low_precision": ".low_precision",
    "exclusion_closure_metadata": ".long_tail_exclusions",
    "load_long_tail_exclusions": ".long_tail_exclusions",
    "low_precision_unvalidated_evidence": ".low_precision",
    "migrate_flashinfer_trace": ".migration",
    "migrate_sol_execbench": ".migration",
    "merge_dataset_shard_traces": ".sharding",
    "normalize_low_precision_format": ".low_precision",
    "plan_dataset_shards": ".sharding",
    "pack_e2m1_codes": ".low_precision",
    "pack_low_precision_tensor": ".low_precision",
    "quantize_e2m1_codes": ".low_precision",
    "build_parity_gap_report": ".parity_gap",
    "build_paper_denominator_report": ".paper_denominator",
    "build_profiler_timing_coverage_report": ".profiler_timing_coverage",
    "inspect_dataset_layout": ".layout",
    "render_parity_gap_markdown": ".parity_gap",
    "render_paper_denominator_markdown": ".paper_denominator",
    "render_profiler_timing_coverage_markdown": ".profiler_timing_coverage",
    "relative_ref": ".evidence_refs",
    "safe_sidecar_stem": ".evidence_refs",
    "sidecar_stem_for_workload": ".evidence_refs",
    "should_skip_cdna4_low_precision_on_arch": ".low_precision",
    "split_excluded_workloads": ".long_tail_exclusions",
    "validate_categories": ".categories",
    "write_dataset_manifest": ".manifest",
    "write_dataset_inventory": ".inventory",
    "write_migration_manifest": ".migration",
    "write_dataset_readiness": ".readiness",
    "write_ready_subset": ".ready_subset",
    "unpack_e2m1_codes": ".low_precision",
    "workload_prefix_lines": ".sharding",
    "workload_shard_paths": ".sharding",
    "write_parity_gap_reports": ".parity_gap",
    "write_paper_denominator_reports": ".paper_denominator",
    "write_profiler_timing_coverage_reports": ".profiler_timing_coverage",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    """Load package exports on first access."""
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Return stable names for interactive discovery and star imports."""
    return sorted({*globals(), *__all__})
