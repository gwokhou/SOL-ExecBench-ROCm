"""Shared Orojenesis adapter exceptions."""

from solar.errors import ToolchainUnavailableError


class OrojenesisError(ToolchainUnavailableError):
    """The official external solver could not produce an auditable bound."""
