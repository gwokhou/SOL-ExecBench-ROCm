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

"""Analyze a SOLAR IR graph into hardware-independent compute and I/O metrics.

This is the second SOLAR pipeline stage, converting an IR graph artifact into
``analysis.yaml``. It emits per-layer and graph totals plus conservative formal
fusion and Orojenesis evidence. See ``SOL_GUIDE.md`` for the memory models and
formal-evidence contract.
"""

from __future__ import annotations

from solar.analysis import formal_evidence
from solar.analysis.graph_context import (
    GraphTopology,
    PreparedAnalysis,
)
from solar.analysis.graph_models import (
    AnalysisAccumulator,
    FormalAnalysis,
    FusionPlan,
    GraphIoTotals,
    LowerBound,
)
from solar.analysis.orojenesis.runner import (
    OrojenesisRunner,
)
from solar.composition import BoundComponent
from solar.schema_versions import OROJENESIS_ANALYSIS_SCHEMA_VERSION
from solar.types import NodeDict


class FormalBoundAnalyzer(BoundComponent):
    """Compute formal lower bounds and proof requirements."""

    def _run_formal_analysis(
        self,
        prepared: PreparedAnalysis,
        topology: GraphTopology,
        io_totals: GraphIoTotals,
        runner: OrojenesisRunner | None,
        *,
        require_orojenesis: bool,
    ) -> FormalAnalysis:
        orojenesis = formal_evidence.new_orojenesis_record(
            semantic_graph=prepared.semantic_graph,
            schema_version=OROJENESIS_ANALYSIS_SCHEMA_VERSION,
        )
        if not (prepared.semantic_graph and prepared.semantic_complete):
            return FormalAnalysis(
                fusion=None,
                orojenesis=orojenesis,
                audited_fused_bytes=io_totals.fused_bytes,
                audited_prefetched_bytes=io_totals.fused_bytes,
                tile_aware_bound=False,
            )
        plan = self._plan_fusion(prepared, topology)
        orojenesis["unsupported_contraction_layers"] = list(
            plan.unsupported_contraction_layers,
        )
        orojenesis["formal_coverage"] = {
            "applicable_layers": 0,
            "total_layers": len(plan.proof_layers)
            + len(plan.unsupported_contraction_layers),
        }
        self._validate_proof_requirements(
            plan,
            runner,
            require_orojenesis=require_orojenesis,
        )
        if runner is not None and plan.proof_layers:
            self._run_orojenesis_evidence(
                plan,
                runner,
                prepared,
                orojenesis,
                require_orojenesis=require_orojenesis,
            )
        elif not plan.proof_layers:
            orojenesis["status"] = formal_evidence.status_without_proof(
                unsupported_contractions=bool(
                    plan.unsupported_contraction_layers,
                ),
                runner_configured=runner is not None,
            )
        audited_prefetched_bytes = io_totals.fused_bytes
        tile_aware_bound = bool(
            require_orojenesis
            and not plan.proof_layers
            and not plan.unsupported_contraction_layers,
        )
        if plan.proof_layers and orojenesis["status"] == "complete":
            candidate_prefetched_bytes, tile_aware_bound = (
                self._audit_orojenesis_evidence(
                    plan,
                    orojenesis,
                    prepared,
                    io_totals.fused_bytes,
                )
            )
            if tile_aware_bound:
                audited_prefetched_bytes = candidate_prefetched_bytes
            else:
                orojenesis["status"] = "incomplete"
        return FormalAnalysis(
            fusion=plan.fusion,
            orojenesis=orojenesis,
            audited_fused_bytes=io_totals.fused_bytes,
            audited_prefetched_bytes=audited_prefetched_bytes,
            tile_aware_bound=tile_aware_bound,
        )

    @staticmethod
    def _validate_proof_requirements(
        plan: FusionPlan,
        runner: OrojenesisRunner | None,
        *,
        require_orojenesis: bool,
    ) -> None:
        if plan.unsupported_contraction_layers and require_orojenesis:
            unsupported = ", ".join(plan.unsupported_contraction_layers)
            raise ValueError(
                "strict formal analysis lacks an exact Orojenesis proof "
                f"representation for contraction layers: {unsupported}",
            )
        if plan.proof_layers and runner is None and require_orojenesis:
            raise ValueError(
                "strict formal analysis requires the pinned Orojenesis toolchain",
            )

    @staticmethod
    def _lower_bound(
        prepared: PreparedAnalysis,
        accumulator: AnalysisAccumulator,
        formal: FormalAnalysis,
        *,
        require_orojenesis: bool,
    ) -> LowerBound:
        seconds: float | None = None
        resource_seconds: dict[str, float] = {}
        compute_resource: str | None = None
        components: NodeDict | None = None
        if (
            prepared.profile is not None
            and prepared.semantic_graph
            and prepared.semantic_complete
        ):
            resource_seconds = prepared.profile.resource_seconds(
                accumulator.resource_work,
            )
            compute_seconds = max(resource_seconds.values(), default=0.0)
            if resource_seconds:
                compute_resource = max(
                    sorted(resource_seconds),
                    key=resource_seconds.__getitem__,
                )
            fused_memory_seconds = (
                formal.audited_fused_bytes
                / prepared.profile.memory_bandwidth_bytes_per_second
            )
            prefetched_memory_seconds = (
                formal.audited_prefetched_bytes
                / prepared.profile.memory_bandwidth_bytes_per_second
            )
            seconds = max(compute_seconds, prefetched_memory_seconds)
            components = {
                "compute_seconds": compute_seconds,
                "resource_seconds": resource_seconds,
                "compute_resource": compute_resource,
                "fused_memory_seconds": fused_memory_seconds,
                "fused_unoverlapped_seconds": compute_seconds
                + fused_memory_seconds,
                "prefetched_memory_seconds": prefetched_memory_seconds,
                "prefetched_overlapped_seconds": seconds,
            }
        if require_orojenesis and not formal.tile_aware_bound:
            raise ValueError(
                "strict analysis did not produce a complete tile-aware bound",
            )
        return LowerBound(
            seconds=seconds,
            resource_seconds=resource_seconds,
            compute_resource=compute_resource,
            components=components,
        )
