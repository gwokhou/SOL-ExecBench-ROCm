"""Ordered expansion rules for complex Torchview operations."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, cast

import networkx as nx

from solar.composition import BoundComponent
from solar.types import NodeDict

type Layer = NodeDict
type LayerMap = dict[str, Layer]
type SourceNodeInfo = NodeDict


@dataclass(frozen=True, slots=True, kw_only=True)
class OperationExpansionRequest:
    """Inputs shared by every complex-operation expansion rule."""

    node_id: str
    node: Layer
    graph: nx.DiGraph
    source_nodes: list[SourceNodeInfo]
    all_source_nodes: list[SourceNodeInfo]
    source_node_ids: dict[str, str]


@dataclass(frozen=True, slots=True, kw_only=True)
class OperationExpansion:
    """Normalized output from one successful expansion rule."""

    layers: LayerMap
    final_node_id: str
    input_mapping: dict[int, str] | None = None


class OperationExpansionHost(Protocol):
    """Converter behavior required by the expansion rule chain."""

    def _should_split_linear_with_bias(self, node_data: Layer) -> bool: ...

    def _split_linear_with_bias(
        self,
        node_id: str,
        node_data: Layer,
        op_graph: nx.DiGraph,
        start_nodes_info: list[SourceNodeInfo],
        start_node_id_map: dict[str, str],
    ) -> tuple[Layer, Layer]: ...

    def _should_expand_groupwise_conv(self, node_data: Layer) -> bool: ...

    def _expand_groupwise_conv(
        self,
        node_id: str,
        node_data: Layer,
        op_graph: nx.DiGraph,
        start_nodes_info: list[SourceNodeInfo],
        start_node_id_map: dict[str, str],
    ) -> tuple[LayerMap, str, dict[int, str]]: ...

    def _should_expand_mha(self, node_data: Layer) -> bool: ...

    def _expand_mha(
        self,
        node_id: str,
        node_data: Layer,
        op_graph: nx.DiGraph,
        start_nodes_info: list[SourceNodeInfo],
        start_node_id_map: dict[str, str],
    ) -> tuple[LayerMap, str, dict[int, str]]: ...

    def _should_expand_lstm(self, node_data: Layer) -> bool: ...

    def _expand_lstm(
        self,
        node_id: str,
        node_data: Layer,
        op_graph: nx.DiGraph,
        start_nodes_info: list[SourceNodeInfo],
        start_node_id_map: dict[str, str],
    ) -> tuple[LayerMap, str, dict[int, str]]: ...

    def _should_expand_gru(self, node_data: Layer) -> bool: ...

    def _expand_gru(
        self,
        node_id: str,
        node_data: Layer,
        op_graph: nx.DiGraph,
        start_nodes_info: list[SourceNodeInfo],
        start_node_id_map: dict[str, str],
    ) -> tuple[LayerMap, str, dict[int, str]]: ...

    def _should_expand_sdpa(self, node_data: Layer) -> bool: ...

    def _expand_sdpa(
        self,
        node_id: str,
        node_data: Layer,
        op_graph: nx.DiGraph,
        start_nodes_info: list[SourceNodeInfo],
        start_node_id_map: dict[str, str],
    ) -> tuple[LayerMap, str, dict[int, str]]: ...


type ExpansionRule = Callable[
    [OperationExpansionHost, OperationExpansionRequest],
    OperationExpansion | None,
]


def _linear_bias_rule(
    host: OperationExpansionHost,
    request: OperationExpansionRequest,
) -> OperationExpansion | None:
    if not host._should_split_linear_with_bias(request.node):
        return None
    matmul, add = host._split_linear_with_bias(
        request.node_id,
        request.node,
        request.graph,
        request.all_source_nodes,
        request.source_node_ids,
    )
    add_node_id = f"{request.node_id}.bias_add"
    return OperationExpansion(
        layers={request.node_id: matmul, add_node_id: add},
        final_node_id=add_node_id,
    )


def _subgraph_expansion(
    request: OperationExpansionRequest,
    expand: Callable[
        [
            str,
            Layer,
            nx.DiGraph,
            list[SourceNodeInfo],
            dict[str, str],
        ],
        tuple[LayerMap, str, dict[int, str]],
    ],
) -> OperationExpansion:
    layers, final_node_id, input_mapping = expand(
        request.node_id,
        request.node,
        request.graph,
        request.source_nodes,
        request.source_node_ids,
    )
    return OperationExpansion(
        layers=layers, final_node_id=final_node_id, input_mapping=input_mapping
    )


def _groupwise_conv_rule(
    host: OperationExpansionHost,
    request: OperationExpansionRequest,
) -> OperationExpansion | None:
    if not host._should_expand_groupwise_conv(request.node):
        return None
    return _subgraph_expansion(request, host._expand_groupwise_conv)


def _mha_rule(
    host: OperationExpansionHost,
    request: OperationExpansionRequest,
) -> OperationExpansion | None:
    if not host._should_expand_mha(request.node):
        return None
    return _subgraph_expansion(request, host._expand_mha)


def _lstm_rule(
    host: OperationExpansionHost,
    request: OperationExpansionRequest,
) -> OperationExpansion | None:
    if not host._should_expand_lstm(request.node):
        return None
    return _subgraph_expansion(request, host._expand_lstm)


def _gru_rule(
    host: OperationExpansionHost,
    request: OperationExpansionRequest,
) -> OperationExpansion | None:
    if not host._should_expand_gru(request.node):
        return None
    return _subgraph_expansion(request, host._expand_gru)


def _sdpa_rule(
    host: OperationExpansionHost,
    request: OperationExpansionRequest,
) -> OperationExpansion | None:
    if not host._should_expand_sdpa(request.node):
        return None
    return _subgraph_expansion(request, host._expand_sdpa)


_EXPANSION_RULES: tuple[ExpansionRule, ...] = (
    _linear_bias_rule,
    _groupwise_conv_rule,
    _mha_rule,
    _lstm_rule,
    _gru_rule,
    _sdpa_rule,
)


def expand_operation(
    host: OperationExpansionHost,
    request: OperationExpansionRequest,
) -> OperationExpansion | None:
    """Return the first applicable complex-operation expansion."""
    for rule in _EXPANSION_RULES:
        if expansion := rule(host, request):
            return expansion
    return None


class OperationExpansionEngine(BoundComponent):
    """Apply the ordered complex-operation rule chain for one façade."""

    def _expand_operation(
        self,
        request: OperationExpansionRequest,
    ) -> OperationExpansion | None:
        """Expand one operation using the host's composed strategies."""
        return expand_operation(
            cast("OperationExpansionHost", self._host),
            request,
        )


__all__ = [
    "OperationExpansion",
    "OperationExpansionEngine",
    "OperationExpansionRequest",
    "expand_operation",
]
