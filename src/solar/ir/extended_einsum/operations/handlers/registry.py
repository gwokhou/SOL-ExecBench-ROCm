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

from collections.abc import Mapping
from functools import cache
from types import MappingProxyType
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from solar.ir.extended_einsum.operations.handlers.base import (
        EinsumOp,
        EinsumOpHandler,
    )
    from solar.types import DynamicValue, TensorShapes


class ReadonlyEinsumOpRegistry(Protocol):
    """Minimal immutable lookup boundary consumed by analyzers."""

    def has_handler(self, op_name: str) -> bool:
        """Return whether a handler exists for an operation."""
        ...

    def get_einsum_op(
        self,
        op_name: str,
        shapes: TensorShapes,
        **kwargs: DynamicValue,
    ) -> EinsumOp:
        """Generate a normalized operation through a registered handler."""
        ...


class EinsumOpRegistry:
    """Immutable operation-handler lookup used during analysis."""

    def __init__(self, handlers: Mapping[str, EinsumOpHandler]) -> None:
        """Freeze a normalized handler mapping."""
        self._op_to_handler = MappingProxyType(dict(handlers))

    @property
    def handlers(self) -> Mapping[str, EinsumOpHandler]:
        """Return the immutable normalized handler inventory."""
        return self._op_to_handler

    def get_handler(self, op_name: str) -> EinsumOpHandler | None:
        """Return the registered handler for one normalized operation."""
        return self._op_to_handler.get(op_name.lower())

    def has_handler(self, op_name: str) -> bool:
        """Return whether a handler exists for the operation."""
        return op_name.lower() in self._op_to_handler

    def get_einsum_op(
        self,
        op_name: str,
        shapes: TensorShapes,
        **kwargs: DynamicValue,
    ) -> EinsumOp:
        """Generate an einsum operation through the registered handler."""
        handler = self.get_handler(op_name)
        if handler is None:
            raise ValueError(f"No handler registered for operation: {op_name}")
        return handler.generate_einsum(op_name, shapes, **kwargs)


class EinsumOpRegistryBuilder:
    """Mutable construction boundary for an immutable handler registry."""

    def __init__(self, *, debug: bool = False) -> None:
        """Initialize an empty builder."""
        self.debug = debug
        self._op_to_handler: dict[str, EinsumOpHandler] = {}

    def register_handler(
        self,
        handler_class: type[EinsumOpHandler],
        *,
        replace_ops: frozenset[str] = frozenset(),
    ) -> EinsumOpRegistryBuilder:
        """Register a handler class and return this builder."""
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
        return self

    def build(self) -> EinsumOpRegistry:
        """Freeze the current inventory as an independent registry."""
        return EinsumOpRegistry(self._op_to_handler)


def _register_builtin_handlers(builder: EinsumOpRegistryBuilder) -> None:
    """Register the explicit built-in inventory in precedence order."""
    from solar.ir.extended_einsum.operations.handlers.builtin_handlers import (
        BUILTIN_HANDLER_CLASSES,
        BUILTIN_HANDLER_OVERRIDE_OPS,
    )

    for handler_class in BUILTIN_HANDLER_CLASSES:
        builder.register_handler(
            handler_class,
            replace_ops=BUILTIN_HANDLER_OVERRIDE_OPS.get(
                handler_class,
                frozenset(),
            ),
        )


def build_builtin_registry(*, debug: bool = False) -> EinsumOpRegistry:
    """Build an independent immutable registry of built-in handlers."""
    builder = EinsumOpRegistryBuilder(debug=debug)
    _register_builtin_handlers(builder)
    return builder.build()


@cache
def builtin_einsum_registry() -> EinsumOpRegistry:
    """Return the process-wide immutable built-in registry."""
    return build_builtin_registry()


__all__ = [
    "EinsumOpRegistry",
    "EinsumOpRegistryBuilder",
    "ReadonlyEinsumOpRegistry",
    "build_builtin_registry",
    "builtin_einsum_registry",
]
