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

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from solar.analysis import formal_evidence
from solar.analysis.contraction_proofs import build_orojenesis_proof_graph
from solar.analysis.fusion import FusionPlanner
from solar.analysis.graph_context import (
    GraphTopology,
    PreparedAnalysis,
    product,
)
from solar.analysis.graph_models import (
    FusionPlan,
)
from solar.analysis.mixin_contract import AnalysisMixinContract
from solar.analysis.operand_provenance import (
    contraction_external_source_dtypes,
    contraction_has_region_boundary_proof,
    contraction_operands_are_graph_external,
)
from solar.analysis.orojenesis.multi_einsum import find_multi_einsum_chains
from solar.analysis.orojenesis.regions import find_multi_einsum_regions
from solar.analysis.orojenesis.runner import (
    OrojenesisRunner,
    select_capacity_point,
)
from solar.ir.contracts import layer_operation
from solar.precision import (
    dtype_bytes,
)
from solar.rocm.architecture import ArchitectureProfile, MemoryLevel


def _layer_compulsory_bytes(
    layer: Mapping[str, Any],
    *,
    word_bytes: int,
) -> float:
    names = layer.get("tensor_names") or {}
    shapes = layer.get("tensor_shapes") or {}
    modeled_tensors = {
        str(name): list(shape)
        for side in ("inputs", "outputs")
        for name, shape in zip(
            names.get(side) or [],
            shapes.get(side) or [],
            strict=True,
        )
    }
    return float(
        sum(product(shape) for shape in modeled_tensors.values()) * word_bytes
    )


def _is_zero_excess_compulsory_witness(
    result: Mapping[str, Any],
    point: Mapping[str, Any] | None,
    compulsory_bytes: float,
) -> bool:
    certificate = result.get("optimality_certificate") or {}
    try:
        word_bytes = int(result["word_bits"]) // 8
        certificate_bytes = (
            int(certificate["compulsory_accesses_words"]) * word_bytes
        )
        solver_bytes = float((point or {})["dram_bytes"])
    except (KeyError, TypeError, ValueError):
        return False
    return bool(
        certificate.get("kind") == "selected_capacity_compulsory_witness_v1"
        and certificate.get("scope") == "selected_capacity_only"
        and certificate_bytes == compulsory_bytes == solver_bytes
    )


class OrojenesisEvidenceMixin(AnalysisMixinContract):
    """Collect and audit layer, chain, and region evidence."""

    def _plan_fusion(
        self,
        prepared: PreparedAnalysis,
        topology: GraphTopology,
    ) -> FusionPlan:
        proof_graph_layers, proof_layers, unsupported_contractions = (
            build_orojenesis_proof_graph(
                prepared.all_layers,
                topology.layers,
                analyzer=self.einsum_analyzer,
            )
        )
        chains = find_multi_einsum_chains(proof_graph_layers)
        regions = find_multi_einsum_regions(proof_graph_layers)
        region_paths = [
            path
            for region in regions
            for path in region.get("physical_paths") or []
        ]
        verified_views = {
            str(layer_id)
            for region in regions
            for path in region.get("physical_paths") or []
            for layer_id in path
            if str(
                layer_operation(
                    prepared.all_layers.get(str(layer_id), {}),
                ).get("target", ""),
            )
            in {"view", "transpose", "permute", "squeeze", "unsqueeze"}
        }
        hierarchy = (
            prepared.profile.memory_hierarchy
            if prepared.profile is not None
            else ()
        )
        return FusionPlan(
            fusion=FusionPlanner(
                prepared.graph,
                multi_einsum_chains=[*chains, *region_paths],
                verified_view_nodes=sorted(verified_views),
            ).plan(hierarchy),
            chains=chains,
            regions=regions,
            proof_layers=proof_layers,
            unsupported_contraction_layers=unsupported_contractions,
        )

    @staticmethod
    def _last_cache(profile: ArchitectureProfile | None) -> MemoryLevel | None:
        if profile is None:
            return None
        known = [
            level
            for level in profile.memory_hierarchy
            if level.capacity_bytes is not None and level.name != "vram"
        ]
        if not known:
            return None
        return max(
            known,
            key=lambda level: int(level.capacity_bytes or 0),
        )

    @staticmethod
    def _word_bits(dtypes: list[str], element_size: float) -> int:
        return min(
            (int(dtype_bytes(dtype) * 8) for dtype in dtypes),
            default=int(element_size * 8),
        )

    @staticmethod
    def _select_capacity_and_rewrite_evidence(
        result: dict[str, Any],
        last_cache: MemoryLevel | None,
        require_orojenesis: bool,
        missing_point_error: str,
        evidence_root: Path,
    ) -> None:
        if last_cache is not None:
            point = select_capacity_point(
                result["curve"],
                int(last_cache.capacity_bytes or 0),
            )
            if point is None and require_orojenesis:
                raise ValueError(missing_point_error)
            result["selected_capacity"] = {
                "level": last_cache.name,
                "capacity_bytes": last_cache.capacity_bytes,
                "point": point,
            }
            certificate = result.get("optimality_certificate")
            if isinstance(certificate, dict):
                certificate["capacity_identity"] = {
                    "level": last_cache.name,
                    "capacity_bytes": int(last_cache.capacity_bytes or 0),
                    "scope": "architecture_profile_cache_domain",
                }
        for evidence in result.get("evidence_files", {}).values():
            evidence["path"] = str(evidence_root / str(evidence["path"]))

    def _run_chain_evidence(
        self,
        plan: FusionPlan,
        runner: OrojenesisRunner,
        prepared: PreparedAnalysis,
        last_cache: MemoryLevel | None,
        orojenesis: dict[str, Any],
        require_orojenesis: bool,
    ) -> None:
        for index, layer_ids in enumerate(plan.chains):
            layers = [
                (layer_id, plan.proof_layers[layer_id])
                for layer_id in layer_ids
            ]
            dtypes = [
                str(dtype)
                for _, layer in layers
                for side in ("inputs", "outputs")
                for dtype in (
                    (layer.get("tensor_dtypes") or {}).get(side) or []
                )
            ]
            chain_id = f"chain_{index}"
            result = runner.run_multi_chain(
                layers,
                prepared.output_dir / "orojenesis" / "chains" / chain_id,
                word_bits=self._word_bits(dtypes, prepared.element_size),
            )
            self._select_capacity_and_rewrite_evidence(
                result,
                last_cache,
                require_orojenesis,
                "multi-einsum Orojenesis produced no point within "
                f"{last_cache.name if last_cache else '<cache>'} capacity for {chain_id}",
                Path("orojenesis") / "chains" / chain_id,
            )
            orojenesis["chains"][chain_id] = result

    def _run_region_evidence(
        self,
        plan: FusionPlan,
        runner: OrojenesisRunner,
        prepared: PreparedAnalysis,
        last_cache: MemoryLevel | None,
        orojenesis: dict[str, Any],
        require_orojenesis: bool,
    ) -> None:
        for index, problem in enumerate(plan.regions):
            region_id = f"region_{index}"
            layer_ids = [str(item) for item in problem.get("schedule") or []]
            dtypes = [
                str(dtype)
                for layer_id in layer_ids
                for side in ("inputs", "outputs")
                for dtype in (
                    (
                        plan.proof_layers[layer_id].get("tensor_dtypes") or {}
                    ).get(side)
                    or []
                )
            ]
            result = runner.run_multi_region(
                problem,
                prepared.output_dir / "orojenesis" / "regions" / region_id,
                word_bits=self._word_bits(dtypes, prepared.element_size),
            )
            self._select_capacity_and_rewrite_evidence(
                result,
                last_cache,
                require_orojenesis,
                "multi-einsum region produced no point within "
                f"{last_cache.name if last_cache else '<cache>'} capacity for {region_id}",
                Path("orojenesis") / "regions" / region_id,
            )
            orojenesis["regions"][region_id] = result

    def _run_layer_evidence(
        self,
        plan: FusionPlan,
        runner: OrojenesisRunner,
        prepared: PreparedAnalysis,
        last_cache: MemoryLevel | None,
        orojenesis: dict[str, Any],
        require_orojenesis: bool,
    ) -> None:
        multi_member_ids = {
            layer_id for chain in plan.chains for layer_id in chain
        }
        multi_member_ids.update(
            str(layer_id)
            for region in plan.regions
            for layer_id in region.get("schedule") or []
        )
        for layer_id, layer in plan.proof_layers.items():
            if layer_id in multi_member_ids:
                continue
            tensor_dtypes = layer.get("tensor_dtypes") or {}
            dtypes = [
                *(
                    str(dtype)
                    for side in ("inputs", "outputs")
                    for dtype in tensor_dtypes.get(side) or []
                ),
                *contraction_external_source_dtypes(layer, prepared.all_layers),
            ]
            result = runner.run_layer(
                layer,
                prepared.output_dir / "orojenesis" / layer_id,
                word_bits=self._word_bits(dtypes, prepared.element_size),
                selected_capacity_bytes=(
                    int(last_cache.capacity_bytes or 0)
                    if last_cache is not None
                    else None
                ),
            )
            self._select_capacity_and_rewrite_evidence(
                result,
                last_cache,
                require_orojenesis,
                f"Orojenesis produced no point within "
                f"{last_cache.name if last_cache else '<cache>'} capacity for {layer_id}",
                Path("orojenesis") / layer_id,
            )
            orojenesis["layers"][layer_id] = result

    def _run_orojenesis_evidence(
        self,
        plan: FusionPlan,
        runner: OrojenesisRunner,
        prepared: PreparedAnalysis,
        orojenesis: dict[str, Any],
        *,
        require_orojenesis: bool,
    ) -> None:
        orojenesis["status"] = "complete"
        orojenesis["toolchain"] = getattr(runner, "toolchain_identity", None)
        if orojenesis["toolchain"] is None and require_orojenesis:
            raise ValueError(
                "strict formal analysis requires Orojenesis toolchain identity",
            )
        last_cache = self._last_cache(prepared.profile)
        self._run_chain_evidence(
            plan,
            runner,
            prepared,
            last_cache,
            orojenesis,
            require_orojenesis,
        )
        self._run_region_evidence(
            plan,
            runner,
            prepared,
            last_cache,
            orojenesis,
            require_orojenesis,
        )
        self._run_layer_evidence(
            plan,
            runner,
            prepared,
            last_cache,
            orojenesis,
            require_orojenesis,
        )

    @staticmethod
    def _audit_layer_evidence(
        plan: FusionPlan,
        orojenesis: dict[str, Any],
        region_by_layer: dict[str, Any],
        all_layers: dict[str, Any],
    ) -> list[float]:
        excesses: list[float] = []
        for layer_id, result in orojenesis["layers"].items():
            layer = plan.proof_layers[layer_id]
            point = (result.get("selected_capacity") or {}).get("point")
            region = region_by_layer[layer_id]
            external = contraction_operands_are_graph_external(
                layer,
                all_layers,
            )
            region_boundary = contraction_has_region_boundary_proof(
                layer,
                region,
                all_layers,
            )
            word_bytes = int(result["word_bits"]) // 8
            compulsory_bytes = _layer_compulsory_bytes(
                layer,
                word_bytes=word_bytes,
            )
            zero_excess = _is_zero_excess_compulsory_witness(
                result,
                point,
                compulsory_bytes,
            )
            applicable = bool(
                point and (external or region_boundary or zero_excess)
            )
            if external:
                provenance = "graph_input_or_recomputable_preprocess"
                reason = "graph_input_or_recomputable_preprocess_contraction"
            elif region_boundary:
                provenance = (
                    "materialized_region_boundary_and_tile_local_postprocess"
                )
                reason = "materialized_region_boundary_contraction"
            elif zero_excess:
                provenance = "selected_capacity_compulsory_zero_excess"
                reason = "internal_contraction_zero_excess_witness"
            else:
                provenance = "unproven_internal_operand"
                reason = "internal_operand_requires_composition_proof"
            result["formal_applicability"] = {
                "applicable": applicable,
                "region": region["id"],
                "graph_input_operands": external,
                "region_boundary_operands": region_boundary,
                "operand_provenance": provenance,
                "reason": reason,
            }
            if not applicable:
                continue
            if point is None:
                raise ValueError(
                    "applicable layer evidence has no selected point"
                )
            solver_bytes = float(point["dram_bytes"])
            result["audited_dram_bytes"] = solver_bytes
            result["modeled_compulsory_bytes"] = compulsory_bytes
            excesses.append(max(0.0, solver_bytes - compulsory_bytes))
        return excesses

    @staticmethod
    def _audit_chain_evidence(
        plan: FusionPlan,
        orojenesis: dict[str, Any],
        region_by_layer: dict[str, Any],
    ) -> list[float]:
        excesses: list[float] = []
        for result in orojenesis["chains"].values():
            point = (result.get("selected_capacity") or {}).get("point")
            descriptors = (
                (result.get("problem") or {}).get("chain") or {}
            ).get(
                "layers",
            ) or []
            layer_ids = [str(item.get("id")) for item in descriptors]
            region_ids = {
                str(region_by_layer[layer_id]["id"])
                for layer_id in layer_ids
                if layer_id in region_by_layer
            }
            applicable = bool(
                point
                and len(layer_ids) >= 2
                and len(region_ids) == 1
                and all(
                    layer_id in plan.proof_layers for layer_id in layer_ids
                ),
            )
            result["formal_applicability"] = {
                "applicable": applicable,
                "region": next(iter(region_ids), None),
                "layer_ids": layer_ids,
                "operand_provenance": "graph_inputs_and_internal_chain_edges",
                "reason": (
                    "verified_linear_matmul_tiled_fusion"
                    if applicable
                    else "multi_einsum_chain_or_region_mismatch"
                ),
            }
            if not applicable:
                continue
            if point is None:
                raise ValueError(
                    "applicable chain evidence has no selected point"
                )
            first, last = descriptors[0], descriptors[-1]
            compulsory_elements = int(first["m"]) * int(first["k"])
            compulsory_elements += sum(
                int(item["k"]) * int(item["n"]) for item in descriptors
            )
            compulsory_elements += int(last["m"]) * int(last["n"])
            compulsory_bytes = float(
                compulsory_elements * (int(result["word_bits"]) // 8),
            )
            solver_bytes = float(point["dram_bytes"])
            result["audited_dram_bytes"] = solver_bytes
            result["modeled_compulsory_bytes"] = compulsory_bytes
            excesses.append(max(0.0, solver_bytes - compulsory_bytes))
        return excesses

    @staticmethod
    def _audit_region_evidence(
        plan: FusionPlan,
        orojenesis: dict[str, Any],
        region_by_layer: dict[str, Any],
    ) -> list[float]:
        excesses: list[float] = []
        for result in orojenesis["regions"].values():
            point = (result.get("selected_capacity") or {}).get("point")
            problem = result.get("problem") or {}
            descriptors = problem.get("nodes") or []
            layer_ids = [str(item.get("id")) for item in descriptors]
            region_ids = {
                str(region_by_layer[layer_id]["id"])
                for layer_id in layer_ids
                if layer_id in region_by_layer
            }
            applicable = bool(
                point
                and len(layer_ids) >= 2
                and len(region_ids) == 1
                and all(
                    layer_id in plan.proof_layers for layer_id in layer_ids
                ),
            )
            result["formal_applicability"] = {
                "applicable": applicable,
                "region": next(iter(region_ids), None),
                "layer_ids": layer_ids,
                "operand_provenance": (
                    "graph_inputs_and_verified_internal_region_edges"
                ),
                "reason": (
                    "verified_matmul_region_tiled_fusion"
                    if applicable
                    else "multi_einsum_region_or_fusion_mismatch"
                ),
            }
            if not applicable:
                continue
            if point is None:
                raise ValueError(
                    "applicable region evidence has no selected point"
                )
            by_id = {str(item["id"]): item for item in descriptors}
            roots = [str(item) for item in problem.get("roots") or []]
            leaves = [str(item) for item in problem.get("leaves") or []]
            compulsory_elements = sum(
                int(by_id[root]["m"]) * int(by_id[root]["k"]) for root in roots
            )
            compulsory_elements += sum(
                int(item["k"]) * int(item["n"]) for item in descriptors
            )
            compulsory_elements += sum(
                int(by_id[leaf]["m"]) * int(by_id[leaf]["n"]) for leaf in leaves
            )
            compulsory_bytes = float(
                compulsory_elements * (int(result["word_bits"]) // 8),
            )
            solver_bytes = float(point["dram_bytes"])
            result["audited_dram_bytes"] = solver_bytes
            result["modeled_compulsory_bytes"] = compulsory_bytes
            excesses.append(max(0.0, solver_bytes - compulsory_bytes))
        return excesses

    def _audit_orojenesis_evidence(
        self,
        plan: FusionPlan,
        orojenesis: dict[str, Any],
        prepared: PreparedAnalysis,
        audited_fused_bytes: float,
    ) -> tuple[float, bool]:
        region_by_layer = {
            layer_id: region
            for region in plan.fusion["regions"]
            for layer_id in region["layers"]
        }
        excesses = self._audit_layer_evidence(
            plan,
            orojenesis,
            region_by_layer,
            prepared.all_layers,
        )
        excesses.extend(
            self._audit_chain_evidence(plan, orojenesis, region_by_layer),
        )
        excesses.extend(
            self._audit_region_evidence(plan, orojenesis, region_by_layer),
        )
        tile_aware_bound = formal_evidence.audit_tile_evidence_contract(
            orojenesis,
            evidence_root=prepared.output_dir,
            proof_layer_count=len(plan.proof_layers),
            unsupported_layer_count=len(plan.unsupported_contraction_layers),
        )
        return audited_fused_bytes + max(
            excesses,
            default=0.0,
        ), tile_aware_bound
