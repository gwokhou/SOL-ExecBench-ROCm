"""Runtime enforcement for statically typed mixin composition contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod, update_abstractmethods
from collections.abc import Callable, Iterable


def runtime_mixin_contract(
    name: str,
    required_members: Iterable[str],
) -> type[ABC]:
    """Create an abstract runtime counterpart for a type-checking protocol."""

    class RuntimeMixinContract(ABC):  # noqa: B024 -- populated dynamically
        """Base populated with abstract members below."""

    RuntimeMixinContract.__name__ = name
    RuntimeMixinContract.__qualname__ = name
    for member_name in required_members:
        setattr(
            RuntimeMixinContract,
            member_name,
            _abstract_member(name, member_name),
        )
    update_abstractmethods(RuntimeMixinContract)
    return RuntimeMixinContract


def _abstract_member(
    contract_name: str,
    member_name: str,
) -> Callable[..., object]:
    @abstractmethod
    def missing_member(*_args: object, **_kwargs: object) -> object:
        raise NotImplementedError(
            f"{contract_name} requires a concrete {member_name} implementation",
        )

    missing_member.__name__ = member_name
    return missing_member


__all__ = ["runtime_mixin_contract"]
