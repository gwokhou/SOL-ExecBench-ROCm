# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Deterministic coverage and differential reports for semantic bound graphs.

The report deliberately compares ``torch.export`` authority capture with the
diagnostic FX/AST path. A diagnostic match does not promote authority, but a
delta is actionable evidence whenever PyTorch or the local fallback changes.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

from sol_execbench.core.data.definition import Definition
from sol_execbench.core.data.workload import Workload
from sol_execbench.core.scoring.amd_bound_graph.builder import (
    build_authority_bound_graph,
    build_bound_graph,
)
from sol_execbench.core.scoring.amd_bound_graph.models import BoundGraph


SEMANTIC_GRAPH_COVERAGE_SCHEMA_VERSION = "sol_execbench.semantic_graph_coverage.v1"


@dataclass(frozen=True)
class SemanticGraphComparison:
    """One workload's authority-capture and diagnostic-graph comparison."""

    definition: str
    workload_uuid: str
    authority_captured: bool
    authority_node_count: int
    diagnostic_node_count: int
    authority_op_families: tuple[str, ...]
    diagnostic_op_families: tuple[str, ...]
    authority_output_metadata_complete: bool
    diagnostic_output_metadata_complete: bool
    differences: tuple[str, ...]
    authority_warnings: tuple[str, ...]
    diagnostic_warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "definition": self.definition,
            "workload_uuid": self.workload_uuid,
            "authority_captured": self.authority_captured,
            "authority_node_count": self.authority_node_count,
            "diagnostic_node_count": self.diagnostic_node_count,
            "authority_op_families": list(self.authority_op_families),
            "diagnostic_op_families": list(self.diagnostic_op_families),
            "authority_output_metadata_complete": self.authority_output_metadata_complete,
            "diagnostic_output_metadata_complete": self.diagnostic_output_metadata_complete,
            "differences": list(self.differences),
            "authority_warnings": list(self.authority_warnings),
            "diagnostic_warnings": list(self.diagnostic_warnings),
        }


@dataclass(frozen=True)
class SemanticGraphCoverageReport:
    """Stable, aggregate view of semantic capture coverage and mismatches."""

    comparisons: tuple[SemanticGraphComparison, ...]

    def to_dict(self) -> dict[str, object]:
        authority_family_counts = Counter(
            family
            for comparison in self.comparisons
            if comparison.authority_captured
            for family in comparison.authority_op_families
        )
        return {
            "schema_version": SEMANTIC_GRAPH_COVERAGE_SCHEMA_VERSION,
            "total_workloads": len(self.comparisons),
            "authority_captured_workloads": sum(
                comparison.authority_captured for comparison in self.comparisons
            ),
            "authority_fallback_workloads": sum(
                not comparison.authority_captured for comparison in self.comparisons
            ),
            "output_metadata_mismatch_workloads": sum(
                comparison.authority_output_metadata_complete
                != comparison.diagnostic_output_metadata_complete
                for comparison in self.comparisons
            ),
            "graph_difference_workloads": sum(
                bool(comparison.differences) for comparison in self.comparisons
            ),
            "authority_op_family_counts": dict(sorted(authority_family_counts.items())),
            "comparisons": [comparison.to_dict() for comparison in self.comparisons],
        }


def compare_semantic_graphs(
    definition: Definition,
    workload: Workload,
    *,
    authority_graph: BoundGraph | None = None,
    diagnostic_graph: BoundGraph | None = None,
) -> SemanticGraphComparison:
    """Compare export authority capture to the diagnostic FX/AST graph.

    Optional graphs make this function useful for stored-artifact differential
    checks without executing user reference code again.
    """
    authority = authority_graph or build_authority_bound_graph(definition, workload)
    diagnostic = diagnostic_graph or build_bound_graph(definition, workload)
    authority_families = tuple(node.op_family.value for node in authority.nodes)
    diagnostic_families = tuple(node.op_family.value for node in diagnostic.nodes)
    authority_captured = _is_complete_export_graph(authority)
    differences: list[str] = []
    if not authority_captured:
        differences.append("authority_export_not_captured")
    if authority_families != diagnostic_families:
        differences.append("op_family_sequence_mismatch")
    if len(authority.nodes) != len(diagnostic.nodes):
        differences.append("node_count_mismatch")
    authority_output_complete = _output_metadata_complete(
        definition, workload, authority
    )
    diagnostic_output_complete = _output_metadata_complete(
        definition, workload, diagnostic
    )
    if authority_output_complete != diagnostic_output_complete:
        differences.append("output_metadata_completeness_mismatch")
    return SemanticGraphComparison(
        definition=definition.name,
        workload_uuid=str(workload.uuid),
        authority_captured=authority_captured,
        authority_node_count=len(authority.nodes),
        diagnostic_node_count=len(diagnostic.nodes),
        authority_op_families=authority_families,
        diagnostic_op_families=diagnostic_families,
        authority_output_metadata_complete=authority_output_complete,
        diagnostic_output_metadata_complete=diagnostic_output_complete,
        differences=tuple(differences),
        authority_warnings=authority.warnings,
        diagnostic_warnings=diagnostic.warnings,
    )


def build_semantic_graph_coverage_report(
    cases: Iterable[tuple[Definition, Workload]],
) -> SemanticGraphCoverageReport:
    """Build a deterministic coverage report for concrete definition/workload pairs."""
    comparisons = tuple(
        sorted(
            (
                compare_semantic_graphs(definition, workload)
                for definition, workload in cases
            ),
            key=lambda item: (item.definition, item.workload_uuid),
        )
    )
    return SemanticGraphCoverageReport(comparisons=comparisons)


def _is_complete_export_graph(graph: BoundGraph) -> bool:
    return (
        bool(graph.nodes)
        and all(
            node.attributes.get("trace_source") == "torch.export"
            for node in graph.nodes
        )
        and "semantic_export_failed" not in graph.warnings
    )


def _output_metadata_complete(
    definition: Definition, workload: Workload, graph: BoundGraph
) -> bool:
    output_shapes = definition.get_output_shapes(workload.axes)
    return all(
        (tensor := graph.tensors.get(f"output:{name}")) is not None
        and tensor.producer_node_id is not None
        and tensor.shape == output_shapes.get(name)
        and tensor.dtype == spec.dtype.value
        for name, spec in definition.outputs.items()
    )
