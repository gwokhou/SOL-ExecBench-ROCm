# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Conservative legality and capacity analysis for SOLAR fusion regions."""

# Fusion is deliberately a single ordered proof pass over the DAG so every
# legal/illegal edge and capacity consequence is emitted together.
# pylint: disable=too-few-public-methods,too-many-return-statements,missing-function-docstring,too-many-locals,too-many-branches,too-many-statements

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import networkx as nx

from solar.analysis.contraction_proofs import requires_tile_evidence
from solar.common.constants import dtype_bytes
from solar.ir.contracts import layer_operation, validate_ir_graph
from solar.rocm.architecture import MemoryLevel


def _product(shape: Sequence[int]) -> int:
    result = 1
    for dimension in shape:
        result *= int(dimension)
    return result


def _tensor_bytes(shape: Sequence[int], dtype: str) -> int:
    return int(_product(shape) * dtype_bytes(dtype))


class _DisjointFusionSets:
    def __init__(
        self,
        nodes: Sequence[str],
        layers: Mapping[str, Mapping[str, Any]],
    ) -> None:
        self.parent = {node: node for node in nodes}
        self.contractions = {
            node: int(requires_tile_evidence(layers[node])) for node in nodes
        }

    def find(self, node: str) -> str:
        while self.parent[node] != node:
            self.parent[node] = self.parent[self.parent[node]]
            node = self.parent[node]
        return node

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        self.parent[right_root] = left_root
        self.contractions[left_root] += self.contractions[right_root]


class FusionPlanner:
    """Build maximal regions while treating unproven fusion as illegal."""

    def __init__(
        self,
        graph: Mapping[str, Any],
        *,
        multi_einsum_chains: Sequence[Sequence[str]] = (),
        verified_view_nodes: Sequence[str] = (),
    ) -> None:
        """Initialize a planner from validated semantic graph metadata."""
        validate_ir_graph(graph)
        self.graph = graph
        self.layers = {
            str(key): value
            for key, value in (graph.get("layers") or {}).items()
            if str(value.get("type", "")).lower() != "start"
        }
        self.multi_einsum_edges = {
            (str(producer), str(consumer))
            for chain in multi_einsum_chains
            for producer, consumer in zip(chain, chain[1:], strict=False)
        }
        self.verified_view_nodes = {str(item) for item in verified_view_nodes}

    def _barrier(self, layer_id: str, layer: Mapping[str, Any]) -> str | None:
        semantic = layer_operation(layer)
        effects = semantic.get("effects") or {}
        target = str(semantic.get("target", ""))
        if layer_id in self.verified_view_nodes:
            if (
                target
                not in {"view", "transpose", "permute", "squeeze", "unsqueeze"}
                or effects.get("mutates")
                or effects.get("atomic")
                or effects.get("opaque_library_call")
            ):
                return "invalid_internal_view_proof"
            return None
        if effects.get("mutates"):
            return "mutation"
        if effects.get("aliases"):
            return "observable_alias"
        if effects.get("atomic"):
            return "atomic"
        if effects.get("opaque_library_call"):
            return "opaque_library_call"
        if target in {
            "sum",
            "mean",
            "prod",
            "amax",
            "amin",
            "argmax",
            "argmin",
            "logsumexp",
        }:
            return "synchronizing_reduction"
        outputs = (layer.get("tensor_names") or {}).get("outputs") or []
        if len(outputs) != 1:
            return "multi_output_not_proven_safe"
        return None

    def plan(self, hierarchy: Sequence[MemoryLevel]) -> dict[str, Any]:
        """Build maximal legal fusion regions under a memory hierarchy."""
        dag = self._dependency_graph()
        decisions, groups = self._fusion_groups(dag)
        producers, consumers, metadata = self._tensor_indices()
        regions = [
            self._region_payload(
                index,
                nodes,
                hierarchy,
                producers,
                consumers,
                metadata,
            )
            for index, nodes in enumerate(groups)
        ]
        return {"decisions": decisions, "regions": regions}

    def _dependency_graph(self) -> nx.DiGraph:
        dag = nx.DiGraph()
        dag.add_nodes_from(self.layers)
        for layer_id, layer in self.layers.items():
            for consumer in (layer.get("connections") or {}).get(
                "outputs",
            ) or []:
                if consumer in self.layers:
                    dag.add_edge(layer_id, consumer)
        if not nx.is_directed_acyclic_graph(dag):
            raise ValueError("fusion requires an acyclic semantic graph")
        return dag

    def _fusion_groups(
        self,
        dag: nx.DiGraph,
    ) -> tuple[list[dict[str, Any]], list[list[str]]]:
        sets = _DisjointFusionSets(list(dag.nodes), self.layers)
        decisions: list[dict[str, Any]] = []
        for producer, consumer in dag.edges:
            decision_reason = "pure_dependency"
            reason = self._barrier(
                producer,
                self.layers[producer],
            ) or self._barrier(consumer, self.layers[consumer])
            producer_root = sets.find(producer)
            consumer_root = sets.find(consumer)
            if (
                reason is None
                and requires_tile_evidence(self.layers[consumer])
                and not requires_tile_evidence(self.layers[producer])
                and (producer, consumer) not in self.multi_einsum_edges
            ):
                reason = "einsum_operand_producer_boundary"
            if (
                reason is None
                and producer_root != consumer_root
                and sets.contractions[producer_root]
                + sets.contractions[consumer_root]
                > 1
            ):
                if (producer, consumer) in self.multi_einsum_edges:
                    decision_reason = "verified_multi_einsum_chain"
                else:
                    reason = "multiple_einsums_require_multi_einsum_solver"
            if reason is None:
                sets.union(producer, consumer)
            decisions.append(
                {
                    "producer": producer,
                    "consumer": consumer,
                    "legal": reason is None,
                    "reason": reason or decision_reason,
                },
            )
        groups: dict[str, list[str]] = defaultdict(list)
        for node in nx.topological_sort(dag):
            groups[sets.find(node)].append(node)
        return decisions, list(groups.values())

    def _tensor_indices(
        self,
    ) -> tuple[
        dict[str, str],
        dict[str, set[str]],
        dict[str, tuple[list[int], str]],
    ]:
        producers: dict[str, str] = {}
        consumers: dict[str, set[str]] = defaultdict(set)
        metadata: dict[str, tuple[list[int], str]] = {}
        for node, layer in self.layers.items():
            names = layer.get("tensor_names") or {}
            shapes = layer.get("tensor_shapes") or {}
            dtypes = layer.get("tensor_dtypes") or {}
            for name, shape, dtype in zip(
                names.get("outputs") or [],
                shapes.get("outputs") or [],
                dtypes.get("outputs") or [],
                strict=True,
            ):
                producers[str(name)] = node
                metadata[str(name)] = (list(shape), str(dtype))
            for name, shape, dtype in zip(
                names.get("inputs") or [],
                shapes.get("inputs") or [],
                dtypes.get("inputs") or [],
                strict=True,
            ):
                consumers[str(name)].add(node)
                metadata.setdefault(str(name), (list(shape), str(dtype)))
        return producers, consumers, metadata

    def _region_payload(
        self,
        index: int,
        nodes: list[str],
        hierarchy: Sequence[MemoryLevel],
        producers: Mapping[str, str],
        consumers: Mapping[str, set[str]],
        metadata: Mapping[str, tuple[list[int], str]],
    ) -> dict[str, Any]:
        external_inputs, external_outputs, internal, unfused_bytes = (
            self._region_io(nodes, producers, consumers)
        )
        fused_bytes = sum(
            _tensor_bytes(*metadata[name])
            for name in external_inputs | external_outputs
            if name in metadata
        )
        peak_live = self._peak_live_bytes(
            nodes,
            internal,
            producers,
            consumers,
            metadata,
        )
        return {
            "id": f"region_{index}",
            "layers": nodes,
            "external_inputs": sorted(external_inputs),
            "external_outputs": sorted(external_outputs),
            "unfused_bytes": unfused_bytes,
            "fused_bytes": fused_bytes,
            "prefetched_bytes": fused_bytes,
            "peak_live_bytes": peak_live,
            "capacity": _capacity_evidence(hierarchy, peak_live),
        }

    def _region_io(
        self,
        nodes: list[str],
        producers: Mapping[str, str],
        consumers: Mapping[str, set[str]],
    ) -> tuple[set[str], set[str], set[str], int]:
        node_set = set(nodes)
        external_inputs: set[str] = set()
        external_outputs: set[str] = set()
        internal: set[str] = set()
        unfused_bytes = 0
        for node in nodes:
            layer = self.layers[node]
            shapes = layer.get("tensor_shapes") or {}
            dtypes = layer.get("tensor_dtypes") or {}
            unfused_bytes += sum(
                _tensor_bytes(shape, str(dtype))
                for side in ("inputs", "outputs")
                for shape, dtype in zip(
                    shapes.get(side) or [],
                    dtypes.get(side) or [],
                    strict=True,
                )
            )
            for name in (layer.get("tensor_names") or {}).get("inputs") or []:
                if producers.get(str(name)) not in node_set:
                    external_inputs.add(str(name))
            for name in (layer.get("tensor_names") or {}).get("outputs") or []:
                output_name = str(name)
                uses = consumers.get(output_name) or set()
                if uses and uses.issubset(node_set):
                    internal.add(output_name)
                else:
                    external_outputs.add(output_name)
        return external_inputs, external_outputs, internal, unfused_bytes

    @staticmethod
    def _peak_live_bytes(
        nodes: list[str],
        internal: set[str],
        producers: Mapping[str, str],
        consumers: Mapping[str, set[str]],
        metadata: Mapping[str, tuple[list[int], str]],
    ) -> int:
        positions = {node: offset for offset, node in enumerate(nodes)}
        events: dict[int, int] = defaultdict(int)
        node_set = set(nodes)
        for name in internal:
            size = _tensor_bytes(*metadata[name])
            start = positions[producers[name]]
            end = max(
                positions[consumer]
                for consumer in consumers[name]
                if consumer in node_set
            )
            events[start] += size
            events[end + 1] -= size
        live = peak = 0
        for offset in range(len(nodes) + 1):
            live += events[offset]
            peak = max(peak, live)
        return peak


def _capacity_evidence(
    hierarchy: Sequence[MemoryLevel],
    peak_live_bytes: int,
) -> dict[str, Any]:
    capacities: dict[str, Any] = {}
    for level in hierarchy:
        capacity = level.capacity_bytes
        capacities[level.name] = {
            "scope": level.scope,
            "capacity_bytes": capacity,
            "peak_live_bytes": peak_live_bytes,
            "capacity_pressure_bytes": (
                None if capacity is None else max(0, peak_live_bytes - capacity)
            ),
            "source": level.source,
        }
    return capacities


__all__ = ["FusionPlanner"]
