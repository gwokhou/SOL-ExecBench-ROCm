"""Shared Orojenesis adapter exceptions."""

from solar.errors import (
    MAPPER_TIMEOUT_REASON_CODE,
    MapperTimeoutError,
    ToolchainUnavailableError,
)


class OrojenesisError(ToolchainUnavailableError):
    """The official external solver could not produce an auditable bound."""


class OrojenesisTimeoutError(OrojenesisError, MapperTimeoutError):
    """The pinned Orojenesis mapper exceeded its configured time budget."""

    reason_code = MAPPER_TIMEOUT_REASON_CODE
