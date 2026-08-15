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

"""Convert PyTorch computation graphs to einsum representation.

This module implements the first stage of the Solar pipeline:

    pytorch_graph.yaml -> einsum_graph.yaml -> einsum_graph_renamed.yaml

The output follows the einsum graph schema:

    layers:
      <layer_id>:
        type: <operation_type>
        einsum_equation: <equation_string>
        elementwise_op: <op>
        reduction_op: <op>
        is_real_einsum: <bool>
        is_einsum_supportable: <bool>
        tensor_names: {inputs: [...], outputs: [...]}
        tensor_shapes: {inputs: [...], outputs: [...]}
        connections: {inputs: [...], outputs: [...]}

Example:
    >>> from solar.ir.extended_einsum.torchview.converter import PyTorchToEinsum
    >>> converter = PyTorchToEinsum()
    >>> result = converter.convert("input/pytorch_graph.yaml", "output/")
"""

from __future__ import annotations

from threading import RLock
from typing import ClassVar

from solar.composition import BoundComponent, component_attribute
from solar.ir.extended_einsum.operations.analyzer import EinsumAnalyzer
from solar.ir.extended_einsum.torchview.converter_attention import (
    AttentionOperationConverter,
)
from solar.ir.extended_einsum.torchview.converter_convolution import (
    ConvolutionOperationConverter,
)
from solar.ir.extended_einsum.torchview.converter_expansion import (
    OperationExpansionEngine,
)
from solar.ir.extended_einsum.torchview.converter_layers import (
    GeneralOperationConverter,
)
from solar.ir.extended_einsum.torchview.converter_models import (
    ConversionConfig,
    ConversionState,
    PathLike,
)
from solar.ir.extended_einsum.torchview.converter_normalization import (
    GraphNormalizer,
)
from solar.ir.extended_einsum.torchview.converter_parsing import (
    TorchviewGraphParser,
)
from solar.ir.extended_einsum.torchview.converter_pipeline import (
    GraphEmitter,
)
from solar.ir.extended_einsum.torchview.converter_recurrent import (
    RecurrentOperationConverter,
)
from solar.types import DynamicValue


class PyTorchToEinsum:
    """Convert PyTorch computation graphs to einsum representation.

    This converter transforms pytorch_graph.yaml files into einsum_graph.yaml
    files, translating PyTorch operations into einsum notation where possible.

    Attributes:
        debug: Whether to print debug information.
        enable_agent: Whether to use LLM agent for unknown operations.
        api_key: API key for LLM agent.
        cache_dir: Directory for caching generated handlers.
    """

    def __init__(
        self,
        debug: bool = False,
        enable_agent: bool = False,
        api_key: str | None = None,
        cache_dir: str = "./solar_handlers_cache",
        strict: bool = False,
    ) -> None:
        """Initialize the converter.

        Args:
            debug: Enable debug output.
            enable_agent: Enable LLM agent for unknown node types.
            api_key: OpenAI API key for LLM agent.
            cache_dir: Directory for caching generated handlers.
            strict: Reject unsupported operations instead of passing through.
        """
        self._config = ConversionConfig(
            debug=debug,
            enable_agent=enable_agent,
            api_key=api_key,
            cache_dir=cache_dir,
            strict=strict,
        )
        self._debug = self._config.debug
        self._enable_agent = self._config.enable_agent
        self._api_key = self._config.api_key
        self._cache_dir = self._config.cache_dir
        self._strict = self._config.strict
        self._state = ConversionState()
        self._conversion_lock = RLock()
        self._einsum_analyzer = EinsumAnalyzer(debug=debug)
        self._components: tuple[BoundComponent, ...] = (
            TorchviewGraphParser(self),
            GraphEmitter(self),
            OperationExpansionEngine(self),
            AttentionOperationConverter(self),
            ConvolutionOperationConverter(self),
            RecurrentOperationConverter(self),
            GraphNormalizer(self),
            GeneralOperationConverter(self),
        )

    def __getattr__(self, name: str) -> DynamicValue:
        """Resolve private conversion behavior from composed components."""
        return component_attribute(self._components, name)

    @property
    def _tensor_to_producer_op(self) -> dict[str, str]:
        return self._state.tensor_to_producer

    @_tensor_to_producer_op.setter
    def _tensor_to_producer_op(self, value: dict[str, str]) -> None:
        self._state.tensor_to_producer = value

    @property
    def _tensor_to_producer_slot(self) -> dict[str, int]:
        return self._state.tensor_to_producer_slot

    @_tensor_to_producer_slot.setter
    def _tensor_to_producer_slot(self, value: dict[str, int]) -> None:
        self._state.tensor_to_producer_slot = value

    def convert(
        self,
        pytorch_graph_path: PathLike,
        output_dir: PathLike,
        *,
        copy_graph: bool = True,
        expand_complex_ops: bool = True,
        enable_rename: bool = False,
    ) -> dict[str, DynamicValue] | None:
        """Serialize reuse of this façade and run the composed pipeline."""
        with self._conversion_lock:
            self._state = ConversionState()
            emitter = next(
                component
                for component in self._components
                if isinstance(component, GraphEmitter)
            )
            return emitter.convert(
                pytorch_graph_path,
                output_dir,
                copy_graph=copy_graph,
                expand_complex_ops=expand_complex_ops,
                enable_rename=enable_rename,
            )

    @property
    def debug(self) -> bool:
        """Whether debug output is enabled."""
        return self._debug

    @property
    def einsum_analyzer(self) -> EinsumAnalyzer:
        """The einsum analyzer instance."""
        return self._einsum_analyzer

    _SHAPE_OP_TYPES_FOR_DTYPE: ClassVar[set[str]] = {
        "view",
        "reshape",
        "flatten",
        "unflatten",
        "squeeze",
        "unsqueeze",
        "expand",
        "repeat",
        "repeat_interleave",
        "transpose",
        "permute",
        "t",
        "contiguous",
        "cat",
        "concat",
        "stack",
        "split",
        "chunk",
        "__getitem__",
        "getitem",
        "select",
        "index_select",
        "narrow",
        "slice",
        "movedim",
        "swapaxes",
        "swapdims",
        "view_as",
        "reshape_as",
        "broadcast_to",
        "expand_as",
        "detach",
        "alias",
        "ravel",
        "unbind",
        "diagonal",
    }
    _DTYPE_BITS: ClassVar[dict[str, int]] = {
        "bool": 1,
        "byte": 8,
        "uint8": 8,
        "int8": 8,
        "short": 16,
        "int16": 16,
        "half": 16,
        "float16": 16,
        "bfloat16": 16,
        "int": 32,
        "int32": 32,
        "tf32": 32,
        "float32": 32,
        "long": 64,
        "int64": 64,
        "complex64": 64,
        "double": 64,
        "float64": 64,
        "complex128": 128,
    }
    _OUTPUT_DTYPE_INPUT_INDEX: ClassVar[dict[str, int]] = {
        "cross_entropy": 0,
        "embedding": 1,
        "embedding_bag": 1,
        "gather": 0,
        "index_add": 0,
        "index_add_": 0,
        "index_copy": 0,
        "index_copy_": 0,
        "index_put": 0,
        "index_put_": 0,
        "kl_div": 0,
        "nll_loss": 0,
        "scatter": 0,
        "scatter_": 0,
        "scatter_add": 0,
        "scatter_add_": 0,
        "where": 1,
        "xlogy": 0,
    }
    _PARAMETER_TENSOR_INDICES: ClassVar[dict[str, set[int]]] = {
        "batch_norm": {1, 2, 3, 4},
        "conv1d": {1, 2},
        "conv2d": {1, 2},
        "conv3d": {1, 2},
        "conv_transpose1d": {1, 2},
        "conv_transpose2d": {1, 2},
        "conv_transpose3d": {1, 2},
        "embedding": {1},
        "group_norm": {1, 2},
        "layer_norm": {1, 2},
        "linear": {1, 2},
    }


__all__ = ["PyTorchToEinsum"]
