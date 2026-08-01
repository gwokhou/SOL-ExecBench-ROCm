"""Type contract shared by the Torchview processor mixins."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Protocol

from torch import nn

from solar.graph.torchview.models import NodeInfo
from solar.types import DynamicValue

if TYPE_CHECKING:

    class TorchviewProcessorContract(Protocol):
        """Members supplied by the final ``TorchviewProcessor`` composition."""

        debug: bool
        _VALID_NODE_TYPES: ClassVar[tuple[str, ...]]
        _matched_modules: set[str]
        _node_counter: dict[str, int]
        _original_to_clean_id: dict[str, str]
        _original_to_hierarchical: dict[str, str]
        _processed_nodes: set[str]

        def _extract_from_edge_list(
            self,
            computation_graph: DynamicValue,
            original_model: nn.Module | None = None,
        ) -> list[NodeInfo]: ...

        def _extract_from_visual_graph(
            self,
            visual_graph: DynamicValue,
        ) -> list[NodeInfo]: ...

        def _extract_node_info(
            self,
            node: DynamicValue,
            node_id: str,
            original_model: nn.Module | None = None,
        ) -> NodeInfo: ...

        def _get_pytorch_module(
            self,
            node: DynamicValue,
        ) -> nn.Module | None: ...

        def _extract_module_arguments(
            self,
            module: nn.Module,
        ) -> dict[str, DynamicValue]: ...

        def _apply_model_parameters(
            self,
            layer_nodes: list[NodeInfo],
            model: nn.Module,
            computation_nodes: dict[str, DynamicValue] | None = None,
        ) -> None: ...

else:
    from solar.mixin_contracts import runtime_mixin_contract

    TorchviewProcessorContract = runtime_mixin_contract(
        "TorchviewProcessorContract",
        (
            "_extract_from_edge_list",
            "_extract_from_visual_graph",
            "_extract_node_info",
            "_get_pytorch_module",
            "_extract_module_arguments",
            "_apply_model_parameters",
        ),
    )
