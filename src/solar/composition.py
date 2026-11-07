"""Small primitives for composing stateful SOLAR workflow components."""

from __future__ import annotations

from inspect import getattr_static

from solar.types import DynamicValue

_MISSING = object()


class BoundComponent:
    """Delegate component state and cross-component calls to one façade."""

    __slots__ = ("_host",)

    def __init__(self, host: object) -> None:
        """Bind the component to its owning public façade."""
        object.__setattr__(self, "_host", host)

    def __getattr__(self, name: str) -> DynamicValue:
        """Read shared configuration or behavior from the façade."""
        return getattr(self._host, name)

    def __setattr__(self, name: str, value: object) -> None:
        """Write shared state through the façade's state boundary."""
        if name == "_host":
            object.__setattr__(self, name, value)
            return
        setattr(self._host, name, value)


def component_attribute(
    components: tuple[BoundComponent, ...],
    name: str,
) -> DynamicValue:
    """Resolve one declared component member without recursive fallback."""
    for component in components:
        descriptor = getattr_static(type(component), name, _MISSING)
        if descriptor is not _MISSING:
            if isinstance(descriptor, staticmethod):
                return descriptor.__get__(None, type(component))
            if isinstance(descriptor, classmethod):
                return descriptor.__get__(type(component), type(component))
            return descriptor.__get__(component, type(component))
    raise AttributeError(name)


__all__ = ["BoundComponent", "component_attribute"]
