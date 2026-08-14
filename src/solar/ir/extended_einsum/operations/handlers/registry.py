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

"""Registry for einsum operation handlers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from solar.ir.extended_einsum.operations.handlers.base import (
        EinsumOp,
        EinsumOpHandler,
    )
    from solar.types import DynamicValue, TensorShapes


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
            debug: Print handler registration diagnostics.

        """
        self.debug = debug
        self._op_to_handler: dict[str, EinsumOpHandler] = {}

    def register_handler(
        self,
        handler_class: type[EinsumOpHandler],
        *,
        replace_ops: frozenset[str] = frozenset(),
    ) -> None:
        """Register a handler class.

        Args:
            handler_class: Handler class to register.
            replace_ops: Existing operation mappings this class intentionally
                replaces.

        """
        op_keys = tuple(
            op_name.lower() for op_name in handler_class.supported_ops
        )
        if not op_keys:
            raise ValueError(
                f"{handler_class.__name__} must declare supported_ops",
            )
        if len(set(op_keys)) != len(op_keys):
            raise ValueError(
                f"{handler_class.__name__} declares duplicate supported_ops",
            )
        normalized_replacements = frozenset(
            op_name.lower() for op_name in replace_ops
        )
        unknown_replacements = normalized_replacements - set(op_keys)
        if unknown_replacements:
            raise ValueError(
                "replace_ops must be declared by the replacement handler: "
                f"{sorted(unknown_replacements)}",
            )
        conflicts = {
            op_key: self._op_to_handler[op_key]
            for op_key in op_keys
            if op_key in self._op_to_handler
        }
        undeclared_conflicts = set(conflicts) - normalized_replacements
        if undeclared_conflicts:
            details = ", ".join(
                f"{op_key} ({type(conflicts[op_key]).__name__})"
                for op_key in sorted(undeclared_conflicts)
            )
            raise ValueError(
                f"{handler_class.__name__} would replace registered handlers: "
                f"{details}",
            )
        missing_replacements = normalized_replacements - set(conflicts)
        if missing_replacements:
            raise ValueError(
                "replace_ops do not name registered handlers: "
                f"{sorted(missing_replacements)}",
            )

        handler = handler_class()

        # Map each supported operation to this handler
        for op_key in op_keys:
            self._op_to_handler[op_key] = handler
            if self.debug:
                print(
                    f"Registered handler for '{op_key}': {handler_class.__name__}",
                )

    def register(
        self,
        handler_class: type[EinsumOpHandler],
    ) -> type[EinsumOpHandler]:
        """Decorator to register a handler class.

        Usage:
            @registry.register
            class MyHandler(EinsumOpHandler):
                ...
        """
        self.register_handler(handler_class)
        return handler_class

    def get_handler(self, op_name: str) -> EinsumOpHandler | None:
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
        shapes: TensorShapes,
        **kwargs: DynamicValue,
    ) -> EinsumOp:
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


@dataclass(slots=True)
class _RegistryState:
    """Hold the optional process-wide registry behind the public accessor."""

    registry: EinsumOpRegistry | None = None
    handlers_loaded: bool = False


_REGISTRY_STATE = _RegistryState()


def _register_builtin_handlers(registry: EinsumOpRegistry) -> None:
    """Register the explicit built-in inventory in precedence order."""
    from solar.ir.extended_einsum.operations.handlers.builtin_handlers import (
        BUILTIN_HANDLER_CLASSES,
        BUILTIN_HANDLER_OVERRIDE_OPS,
    )

    for handler_class in BUILTIN_HANDLER_CLASSES:
        registry.register_handler(
            handler_class,
            replace_ops=BUILTIN_HANDLER_OVERRIDE_OPS.get(
                handler_class,
                frozenset(),
            ),
        )


def build_builtin_registry(*, debug: bool = False) -> EinsumOpRegistry:
    """Build an independent registry containing every built-in handler."""
    registry = EinsumOpRegistry(debug=debug)
    _register_builtin_handlers(registry)
    return registry


def get_global_registry(load_handlers: bool = True) -> EinsumOpRegistry:
    """Return the compatibility process-wide registry."""
    if _REGISTRY_STATE.registry is None:
        _REGISTRY_STATE.registry = EinsumOpRegistry()

    if load_handlers and not _REGISTRY_STATE.handlers_loaded:
        _register_builtin_handlers(_REGISTRY_STATE.registry)
        _REGISTRY_STATE.handlers_loaded = True
    return _REGISTRY_STATE.registry


def register_einsum_op(
    handler_class: type[EinsumOpHandler],
) -> type[EinsumOpHandler]:
    """Decorator to register a handler with the global registry.

    Usage:
        @register_einsum_op
        class MyHandler(EinsumOpHandler):
            supported_ops = ["my_op"]
            ...
    """
    get_global_registry(load_handlers=False).register_handler(handler_class)
    return handler_class


__all__ = [
    "EinsumOpRegistry",
    "build_builtin_registry",
    "get_global_registry",
    "register_einsum_op",
]
