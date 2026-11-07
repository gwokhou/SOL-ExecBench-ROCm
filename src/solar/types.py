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

"""Type definitions for the Solar package.

This module defines common types used throughout the Solar package,
following Google's Python style guide for type annotations.
"""

from dataclasses import dataclass, field
from typing import Any

# Type aliases for better readability
DynamicValue = Any
GraphValue = DynamicValue
TensorShape = list[int]
NodeDict = dict[str, Any]


@dataclass(slots=True, kw_only=True)
class TensorShapes:
    """Positional tensor shapes for an operation.

    with ordered lists of shapes matching the einsum operand order.
    """

    inputs: list[TensorShape] = field(default_factory=list)
    outputs: list[TensorShape] = field(default_factory=list)

    @property
    def num_inputs(self) -> int:
        """Return the number of positional input shapes."""
        return len(self.inputs)

    @property
    def num_outputs(self) -> int:
        """Return the number of positional output shapes."""
        return len(self.outputs)

    def input_rank(self, idx: int) -> int:
        """Rank (ndim) of input tensor at position idx."""
        return len(self.inputs[idx]) if idx < len(self.inputs) else 0

    def output_rank(self, idx: int) -> int:
        """Rank (ndim) of output tensor at position idx."""
        return len(self.outputs[idx]) if idx < len(self.outputs) else 0
