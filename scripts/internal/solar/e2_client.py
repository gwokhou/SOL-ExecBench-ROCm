"""Candidate-side observation transport: actual outputs, no oracle or roles."""

import base64
import time
from typing import Any

import torch
from safetensors.torch import save

from scripts.internal.solar.e2_contracts import (
    MAX_OBSERVATION_BYTES,
    OBSERVATION_OPERATION,
)
from sol_execbench.core.bench.reference_protocol import (
    ReferenceClient,
    ReferenceProtocolError,
    ReferenceTimingCase,
    connect_reference_worker,
)
from sol_execbench.core.bench.reward_hack.runtime import (
    _IntegritySnapshot,
    check_runtime_integrity,
)


class ObservationClient(ReferenceClient):
    """Use the existing authenticated channel without extra candidate calls."""

    ipc_seconds: float = 0.0
    integrity_seconds: float = 0.0
    static_review_seconds: float = 0.0

    def check_integrity(
        self, snapshot: _IntegritySnapshot, namespace: dict[str, Any]
    ) -> None:
        """Time the unchanged integrity guard without suppressing its exceptions."""
        start = time.monotonic()
        try:
            check_runtime_integrity(snapshot, namespace)
        finally:
            self.integrity_seconds += time.monotonic() - start

    def _request(self, operation: str, **values: Any) -> ReferenceTimingCase:
        start = time.monotonic()
        try:
            return super()._request(operation, **values)
        finally:
            self.ipc_seconds += time.monotonic() - start

    def validate_timing_outputs(self, outputs: list[torch.Tensor]) -> None:
        """Measure the same incumbent validation IPC, including observer work."""
        start = time.monotonic()
        try:
            super().validate_timing_outputs(outputs)
        finally:
            self.ipc_seconds += time.monotonic() - start

    def observe(
        self,
        row: int,
        round_index: int,
        outputs: list[torch.Tensor],
        checks_seconds: float,
        numeric_passed: bool | None,
    ) -> None:
        """Submit one bounded actual output after the original correctness check."""
        if (
            len(outputs) != 1
            or outputs[0].numel() * outputs[0].element_size()
            > MAX_OBSERVATION_BYTES
        ):
            raise ReferenceProtocolError(
                "judge observation exceeds frozen output budget"
            )
        payload = save({"output": outputs[0].detach().cpu().contiguous()})
        if len(payload) > MAX_OBSERVATION_BYTES:
            raise ReferenceProtocolError(
                "judge observation payload is too large"
            )
        self._request(
            OBSERVATION_OPERATION,
            row_index=row,
            round_index=round_index,
            checks_seconds=checks_seconds,
            numeric_passed=numeric_passed,
            payload=base64.b64encode(payload).decode("ascii"),
        )


def connect_observer(*, device: str) -> ObservationClient:
    """Scrub credentials before candidate import, using the incumbent connector."""
    client = connect_reference_worker(
        device=device, client_type=ObservationClient
    )
    if not isinstance(client, ObservationClient):
        raise TypeError("observation client was not constructed")
    return client
