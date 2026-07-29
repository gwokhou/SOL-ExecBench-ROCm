"""Type contract shared by the Torchview converter mixins."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Protocol

import networkx as nx

from solar.ir.extended_einsum.operations.analyzer import EinsumAnalyzer
from solar.types import DynamicValue

if TYPE_CHECKING:

    class ConverterMixinContract(Protocol):
        """Members supplied by the final ``PyTorchToEinsum`` composition."""

        _DTYPE_BITS: ClassVar[dict[str, int]]
        _OUTPUT_DTYPE_INPUT_INDEX: ClassVar[dict[str, int]]
        _PARAMETER_TENSOR_INDICES: ClassVar[dict[str, set[int]]]
        _SHAPE_OP_TYPES_FOR_DTYPE: ClassVar[set[str]]
        _api_key: str | None
        _cache_dir: str
        _debug: bool
        _einsum_analyzer: EinsumAnalyzer
        _enable_agent: bool
        _strict: bool
        _tensor_to_producer_op: dict[str, str]
        _tensor_to_producer_slot: dict[str, int]

        @staticmethod
        def _as_list(
            value: DynamicValue, default: list[int]
        ) -> list[DynamicValue]: ...

        def _parse_einsum_from_raw_attributes(
            self,
            module_args: dict[str, DynamicValue],
        ) -> str | None: ...

        def _parse_reduction_args_from_raw_attributes(
            self,
            module_args: dict[str, DynamicValue],
        ) -> tuple[list[int] | None, bool]: ...

        def _repair_torchview_quirks(
            self,
            layers: dict[str, DynamicValue],
            op_ids: list[str],
            tensor_ids: list[str],
        ) -> None: ...

        def _validate_tensor_shape_consistency(
            self,
            einsum_graph: dict[str, DynamicValue],
        ) -> None: ...

        def _find_entry_node_for_predecessor(
            self,
            result: dict[str, DynamicValue],
            predecessor_id: str,
            original_node_id: str,
            input_mapping: dict[int, str],
        ) -> str: ...

        def _add_start_nodes(
            self,
            result: dict[str, DynamicValue],
            start_nodes_info: list[dict[str, DynamicValue]],
        ) -> dict[str, str]: ...

        def _validate_input_types_alignment(
            self,
            node_id: str,
            node_data: dict[str, DynamicValue],
        ) -> None: ...

        def _should_split_linear_with_bias(
            self,
            node_data: dict[str, DynamicValue],
        ) -> bool: ...

        def _split_linear_with_bias(
            self,
            node_id: str,
            node_data: dict[str, DynamicValue],
            op_graph: nx.DiGraph,
            start_nodes_info: list[dict[str, DynamicValue]],
            start_node_id_map: dict[str, str],
        ) -> tuple[dict[str, DynamicValue], dict[str, DynamicValue]]: ...

        def _should_expand_groupwise_conv(
            self,
            node_data: dict[str, DynamicValue],
        ) -> bool: ...

        def _expand_groupwise_conv(
            self,
            node_id: str,
            node_data: dict[str, DynamicValue],
            op_graph: nx.DiGraph,
            start_nodes_info: list[dict[str, DynamicValue]],
            start_node_id_map: dict[str, str],
        ) -> tuple[dict[str, dict[str, DynamicValue]], str, dict[int, str]]: ...

        def _should_expand_mha(
            self, node_data: dict[str, DynamicValue]
        ) -> bool: ...

        def _should_expand_sdpa(
            self, node_data: dict[str, DynamicValue]
        ) -> bool: ...

        def _should_expand_lstm(
            self, node_data: dict[str, DynamicValue]
        ) -> bool: ...

        def _should_expand_gru(
            self, node_data: dict[str, DynamicValue]
        ) -> bool: ...

        def _expand_sdpa(
            self,
            node_id: str,
            node_data: dict[str, DynamicValue],
            op_graph: nx.DiGraph,
            start_nodes_info: list[dict[str, DynamicValue]],
            start_node_id_map: dict[str, str],
        ) -> tuple[dict[str, dict[str, DynamicValue]], str, dict[int, str]]: ...

        def _expand_mha(
            self,
            node_id: str,
            node_data: dict[str, DynamicValue],
            op_graph: nx.DiGraph,
            start_nodes_info: list[dict[str, DynamicValue]],
            start_node_id_map: dict[str, str],
        ) -> tuple[dict[str, dict[str, DynamicValue]], str, dict[int, str]]: ...

        def _expand_lstm(
            self,
            node_id: str,
            node_data: dict[str, DynamicValue],
            op_graph: nx.DiGraph,
            start_nodes_info: list[dict[str, DynamicValue]],
            start_node_id_map: dict[str, str],
        ) -> tuple[dict[str, dict[str, DynamicValue]], str, dict[int, str]]: ...

        def _expand_gru(
            self,
            node_id: str,
            node_data: dict[str, DynamicValue],
            op_graph: nx.DiGraph,
            start_nodes_info: list[dict[str, DynamicValue]],
            start_node_id_map: dict[str, str],
        ) -> tuple[dict[str, dict[str, DynamicValue]], str, dict[int, str]]: ...

        def _convert_operation(
            self,
            node_id: str,
            node_data: dict[str, DynamicValue],
            op_graph: nx.DiGraph,
            start_nodes_info: list[dict[str, DynamicValue]],
            start_node_id_map: dict[str, str],
        ) -> dict[str, DynamicValue]: ...

        def _fix_split_connections(
            self,
            result: dict[str, DynamicValue],
            node_id_remap: dict[str, str],
            expanded_input_map: dict[str, dict[int, str]] | None = None,
        ) -> None: ...

else:

    class ConverterMixinContract:
        """Runtime-empty base for Torchview converter mixins."""
