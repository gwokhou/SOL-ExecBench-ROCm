"""Type contract shared by the graph-analysis mixins."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from solar.analysis.graph_context import GraphTopology, PreparedAnalysis
from solar.analysis.graph_models import (
    FusionPlan,
    LayerCompute,
    LayerData,
    MemoryBytes,
    MemoryElements,
)
from solar.analysis.graph_validation import GraphValidator
from solar.analysis.orojenesis.runner import OrojenesisRunner
from solar.ir.extended_einsum.operations.analyzer import EinsumAnalyzer
from solar.types import DynamicValue

if TYPE_CHECKING:

    class AnalysisMixinContract(Protocol):
        """Members supplied by the final ``IRGraphAnalyzer`` composition."""

        debug: bool
        validator: GraphValidator
        einsum_analyzer: EinsumAnalyzer

        @staticmethod
        def _parse_layer(
            layer_id: str,
            layer: dict[str, DynamicValue],
        ) -> LayerData: ...

        def _compute_layer(
            self,
            data: LayerData,
            *,
            strict: bool,
        ) -> LayerCompute: ...

        @staticmethod
        def _memory_elements(
            data: LayerData,
            compute: LayerCompute,
            topology: GraphTopology,
        ) -> MemoryElements: ...

        @staticmethod
        def _memory_bytes(
            data: LayerData,
            memory: MemoryElements,
            fallback_precision: str,
        ) -> MemoryBytes: ...

        def _plan_fusion(
            self,
            prepared: PreparedAnalysis,
            topology: GraphTopology,
        ) -> FusionPlan: ...

        def _run_orojenesis_evidence(
            self,
            plan: FusionPlan,
            runner: OrojenesisRunner,
            prepared: PreparedAnalysis,
            orojenesis: dict[str, DynamicValue],
            *,
            require_orojenesis: bool,
        ) -> None: ...

        def _audit_orojenesis_evidence(
            self,
            plan: FusionPlan,
            orojenesis: dict[str, DynamicValue],
            prepared: PreparedAnalysis,
            audited_fused_bytes: float,
        ) -> tuple[float, bool]: ...

else:

    class AnalysisMixinContract:
        """Runtime-empty base for graph-analysis mixins."""
