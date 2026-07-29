# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Typed intermediate nodes produced by the Torchview extractor."""

from dataclasses import dataclass, field
from typing import Any

from solar.types import NodeDict, TensorShape


@dataclass
class NodeInfo:
    """Normalized metadata for one Torchview computation node."""

    node_id: str
    type: str
    node_class: str = "UnknownNode"
    input_nodes: list[str] = field(default_factory=list)
    output_nodes: list[str] = field(default_factory=list)
    input_shapes: list[TensorShape] = field(default_factory=list)
    output_shapes: list[TensorShape] = field(default_factory=list)
    input_dtypes: list[str] = field(default_factory=list)
    output_dtypes: list[str] = field(default_factory=list)
    input_types: list[str] = field(default_factory=list)
    output_types: list[str] = field(default_factory=list)
    output_slots: list[dict[str, Any]] = field(default_factory=list)
    module_args: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> NodeDict:
        """Return the canonical operator-graph node mapping."""
        return {
            "type": self.type,
            "node_class": self.node_class,
            "input_shapes": self.input_shapes,
            "output_shapes": self.output_shapes,
            "input_dtypes": self.input_dtypes,
            "output_dtypes": self.output_dtypes,
            "input_types": self.input_types,
            "output_types": self.output_types,
            "output_slots": self.output_slots,
            "module_args": self.module_args,
            "connections": {
                "inputs": self.input_nodes,
                "outputs": self.output_nodes,
            },
        }


__all__ = ["NodeInfo"]
