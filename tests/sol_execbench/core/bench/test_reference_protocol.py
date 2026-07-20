from __future__ import annotations

import json
import os
from contextlib import contextmanager
from multiprocessing.connection import Connection
from typing import Iterator

import pytest
import torch

from sol_execbench.core.bench import reference_protocol
from sol_execbench.core.bench.reference_protocol import (
    PROTOCOL_VERSION,
    ReferenceCase,
    ReferenceExecutionError,
    ReferenceFailureKind,
    ReferenceProtocolError,
    receive_case,
    send_case,
    send_failure,
    send_json,
)


@contextmanager
def _one_way_connection() -> Iterator[tuple[Connection, Connection]]:
    read_fd, write_fd = os.pipe()
    sender = Connection(write_fd, readable=False, writable=True)
    receiver = Connection(read_fd, readable=True, writable=False)
    try:
        yield sender, receiver
    finally:
        sender.close()
        receiver.close()


def test_reference_case_round_trip_uses_safe_tensor_payload() -> None:
    base = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    view = base[:, ::2]
    with _one_way_connection() as (sender, receiver):
        send_case(
            sender,
            ReferenceCase(inputs=[view, 7], outputs=[view + 1]),
            reference_latency_ms=1.25,
        )

        result = receive_case(receiver, device="cpu")

    assert result.reference_latency_ms == 1.25
    assert result.inputs[1] == 7
    assert result.inputs[0].stride() == view.stride()
    assert torch.equal(result.inputs[0], view)
    assert torch.equal(result.outputs[0], view + 1)


def test_reference_failure_is_not_deserialized_as_pickle() -> None:
    with _one_way_connection() as (sender, receiver):
        send_failure(sender, "reference exploded")
        with pytest.raises(ReferenceExecutionError, match="reference exploded") as exc:
            receive_case(receiver, device="cpu")
    assert exc.value.kind is ReferenceFailureKind.REFERENCE_EXECUTION


def test_protocol_version_is_explicit() -> None:
    assert PROTOCOL_VERSION == "sol_execbench.reference_ipc.v1"


def test_invalid_failure_category_is_rejected_as_protocol_error() -> None:
    with _one_way_connection() as (sender, receiver):
        sender.send_bytes(
            json.dumps({"ok": False, "failure_kind": ["not", "a", "string"]}).encode()
        )
        with pytest.raises(ReferenceProtocolError, match="category is invalid"):
            receive_case(receiver, device="cpu")


def test_send_json_wraps_closed_pipe_as_protocol_error() -> None:
    read_fd, write_fd = os.pipe()
    os.close(read_fd)
    sender = Connection(write_fd, readable=False, writable=True)
    try:
        with pytest.raises(ReferenceProtocolError, match="control channel closed"):
            send_json(sender, {"protocol": PROTOCOL_VERSION})
    finally:
        sender.close()


def test_receive_case_rejects_oversized_payload_before_allocation(monkeypatch) -> None:
    monkeypatch.setattr(reference_protocol, "_MAX_PAYLOAD_BYTES", 4)
    with _one_way_connection() as (sender, receiver):
        send_json(
            sender,
            {
                "ok": True,
                "protocol": PROTOCOL_VERSION,
                "inputs": [],
                "outputs": [],
                "payload_bytes": 5,
                "reference_latency_ms": 0.0,
            },
        )
        sender.send_bytes(b"12345")

        with pytest.raises(ReferenceProtocolError, match="payload is too large"):
            receive_case(receiver, device="cpu")


def test_receive_case_wraps_device_materialization_failure(monkeypatch) -> None:
    with _one_way_connection() as (sender, receiver):
        send_case(sender, ReferenceCase(inputs=[torch.ones(1)], outputs=[]))
        monkeypatch.setattr(
            reference_protocol,
            "_decode_values",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                RuntimeError("device out of memory")
            ),
        )

        with pytest.raises(
            ReferenceProtocolError, match="tensor materialization failed"
        ):
            receive_case(receiver, device="cuda")
