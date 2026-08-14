# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Current first-party wire-protocol identifiers.

Protocol identifiers select behavior within an already-defined wire payload.
They are deliberately separate from artifact schema identifiers: using a
protocol does not make the containing Python class or JSON object a new schema
family.
"""

from enum import StrEnum
from typing import Final


class WireProtocol(StrEnum):
    """Current versioned protocols carried by first-party wire contracts."""

    REFERENCE_IPC = "sol_execbench.reference_ipc.v2"
    ROCM_EVENT_TIMING_CUSTOM = "sol_execbench.rocm_event_timing.custom.v4"
    ROCM_EVENT_TIMING_PAPER_COUNTS = (
        "sol_execbench.rocm_event_timing.paper_counts.v4"
    )


CURRENT_WIRE_PROTOCOLS: Final[frozenset[str]] = frozenset(
    protocol.value for protocol in WireProtocol
)

__all__ = ["CURRENT_WIRE_PROTOCOLS", "WireProtocol"]
