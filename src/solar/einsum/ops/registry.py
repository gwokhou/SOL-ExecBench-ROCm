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

"""Registry for einsum operation handlers.

This module provides a centralized registry for managing einsum operation
handlers. Handlers can be registered using a decorator or explicit registration.
"""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from solar.common.types import TensorShapes
    from solar.einsum.ops.base import EinsumOp, EinsumOpHandler

_BUILTIN_HANDLER_MODULES = (
    "attention_ops",
    "conv_ops",
    "cumulative_ops",
    "elementwise_ops",
    "loss_ops",
    "matmul_ops",
    "misc_ops",
    "norm_ops",
    "pooling_ops",
    "reduction_ops",
    "shape_ops",
)


class EinsumOpRegistry:
    """Registry for einsum operation handlers.

    This class manages a collection of EinsumOpHandler instances and provides
    methods for registration and lookup.

    Usage:
        registry = EinsumOpRegistry()

        # Register a handler class
        registry.register_handler(MatmulHandler)

        # Or use the decorator
        @registry.register
        class MyHandler(EinsumOpHandler):
            supported_ops = ["my_op"]
            ...

        # Get einsum for an operation
        einsum_op = registry.get_einsum_op("matmul", shapes)
    """

    def __init__(self, debug: bool = False) -> None:
        """Initialize the registry.

        Args:
            debug: Enable debug output for handlers.

        """
        self.debug = debug
        self._handlers: dict[str, EinsumOpHandler] = {}
        self._handler_classes: list[type[EinsumOpHandler]] = []
        self._op_to_handler: dict[str, EinsumOpHandler] = {}

    def register_handler(self, handler_class: type["EinsumOpHandler"]) -> None:
        """Register a handler class.

        Args:
            handler_class: Handler class to register.

        """
        # Instantiate the handler
        handler = handler_class(debug=self.debug)
        self._handler_classes.append(handler_class)

        # Map each supported operation to this handler
        for op_name in handler.supported_ops:
            op_key = op_name.lower()
            self._op_to_handler[op_key] = handler
            if self.debug:
                print(
                    f"Registered handler for '{op_key}': {handler_class.__name__}",
                )

    def register(
        self,
        handler_class: type["EinsumOpHandler"],
    ) -> type["EinsumOpHandler"]:
        """Decorator to register a handler class.

        Usage:
            @registry.register
            class MyHandler(EinsumOpHandler):
                ...
        """
        self.register_handler(handler_class)
        return handler_class

    def get_handler(self, op_name: str) -> "EinsumOpHandler | None":
        """Get the handler for an operation.

        Args:
            op_name: Operation name.

        Returns:
            Handler if registered, None otherwise.

        """
        return self._op_to_handler.get(op_name.lower())

    def has_handler(self, op_name: str) -> bool:
        """Check if a handler is registered for an operation.

        Args:
            op_name: Operation name.

        Returns:
            True if a handler exists.

        """
        return op_name.lower() in self._op_to_handler

    def get_einsum_op(
        self,
        op_name: str,
        shapes: "TensorShapes",
        **kwargs: Any,
    ) -> "EinsumOp":
        """Get an einsum operation for the given operation name.

        Args:
            op_name: Operation name.
            shapes: Positional input/output tensor shapes.
            **kwargs: Additional operation-specific parameters.

        Returns:
            EinsumOp for the operation.

        Raises:
            ValueError: If no handler is registered for the operation.

        """
        handler = self.get_handler(op_name)
        if handler is None:
            raise ValueError(f"No handler registered for operation: {op_name}")

        return handler.generate_einsum(op_name, shapes, **kwargs)


class _RegistryState:
    """Hold the import-time registry state behind the public accessor."""

    def __init__(self) -> None:
        self.registry: EinsumOpRegistry | None = None
        self.handlers_loaded = False
        self.loading_handlers = False


# Handler decorators run while their modules are imported, so one process-wide
# state object is required to break recursive initialization. Keeping it private
# and exposing it only through get_global_registry constrains mutation.
_REGISTRY_STATE = _RegistryState()


def get_global_registry(load_handlers: bool = True) -> EinsumOpRegistry:
    """Get the global einsum operation registry.

    Args:
        load_handlers: If True, load all handlers on first access.

    Returns:
        The global registry instance.

    """
    if _REGISTRY_STATE.registry is None:
        _REGISTRY_STATE.registry = EinsumOpRegistry()

    # Load handlers if requested and not already loaded/loading
    if (
        load_handlers
        and not _REGISTRY_STATE.handlers_loaded
        and not _REGISTRY_STATE.loading_handlers
    ):
        _REGISTRY_STATE.loading_handlers = True
        try:
            _load_all_handlers()
            _REGISTRY_STATE.handlers_loaded = True
        finally:
            _REGISTRY_STATE.loading_handlers = False

    return _REGISTRY_STATE.registry


def _load_all_handlers() -> None:
    """Load all built-in handlers."""
    for module_name in _BUILTIN_HANDLER_MODULES:
        import_module(f"solar.einsum.ops.{module_name}")


def register_einsum_op(
    handler_class: type["EinsumOpHandler"],
) -> type["EinsumOpHandler"]:
    """Decorator to register a handler with the global registry.

    Usage:
        @register_einsum_op
        class MyHandler(EinsumOpHandler):
            supported_ops = ["my_op"]
            ...
    """
    # Get registry without loading handlers to avoid circular import
    get_global_registry(load_handlers=False).register_handler(handler_class)
    return handler_class


__all__ = ["EinsumOpRegistry", "get_global_registry", "register_einsum_op"]
