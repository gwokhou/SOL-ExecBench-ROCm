# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Authenticated, pickle-free IPC for trusted reference evaluation."""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from enum import StrEnum
from multiprocessing.connection import Connection
from typing import Any

import torch
from safetensors.torch import (
    load as load_safetensors_bytes,
    save as save_safetensors_bytes,
)

from sol_execbench.core.bench.performance_model.access_evidence import (
    AccessPatternSummary,
)
from sol_execbench.core.integrity import sha256_bytes, stable_json_checksum
from sol_execbench.core.integrity.schema_versions import (
    SchemaVersion,
)

REFERENCE_REQUEST_FD_ENV = "SOL_EXECBENCH_REFERENCE_REQUEST_FD"
REFERENCE_RESPONSE_FD_ENV = "SOL_EXECBENCH_REFERENCE_RESPONSE_FD"
REFERENCE_TOKEN_ENV = "SOL_EXECBENCH_REFERENCE_TOKEN"
REFERENCE_PID_ENV = "SOL_EXECBENCH_REFERENCE_PID"
TRUSTED_DEFINITION_FILE = "reference_definition.json"
_MAX_HEADER_BYTES = 1 << 20
MAX_REFERENCE_PAYLOAD_BYTES = 1 << 30
MAX_REFERENCE_TENSOR_STORAGE_BYTES = (
    MAX_REFERENCE_PAYLOAD_BYTES - _MAX_HEADER_BYTES
)


class ReferenceProtocolError(RuntimeError):
    """The trusted reference IPC contract was violated."""


class ReferenceFailureKind(StrEnum):
    """Stable failure categories returned by the trusted reference service."""

    INPUT_GENERATION = "input_generation"
    REFERENCE_EXECUTION = "reference_execution"


class ReferenceExecutionError(RuntimeError):
    """The trusted service could not produce a case for a classified reason."""

    def __init__(self, message: str, *, kind: ReferenceFailureKind) -> None:
        """Initialize a classified trusted-reference failure."""
        super().__init__(message)
        self.kind = kind


@dataclass(frozen=True)
class ReferenceCase:
    """Inputs and expected outputs produced outside the candidate process."""

    inputs: list[Any]
    outputs: list[torch.Tensor]


@dataclass(frozen=True)
class ReferenceTimingCase(ReferenceCase):
    """A fresh timing case with independently measured reference latency."""

    reference_latency_ms: float
    input_sha256: str
    timing_failure: str | None = None
    access_patterns: list[AccessPatternSummary] = field(default_factory=list)


def send_json(connection: Connection, value: dict[str, Any]) -> None:
    """Send a strictly encoded JSON control frame."""
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    if len(encoded) > _MAX_HEADER_BYTES:
        raise ReferenceProtocolError("reference IPC JSON header is too large")
    _send_bytes(connection, encoded, channel="control")


def _send_bytes(
    connection: Connection,
    payload: bytes,
    *,
    channel: str,
) -> None:
    try:
        connection.send_bytes(payload)
    except (EOFError, OSError) as exc:
        raise ReferenceProtocolError(
            f"reference IPC {channel} channel closed",
        ) from exc


def receive_json(connection: Connection) -> dict[str, Any]:
    """Receive a JSON object control frame."""
    try:
        encoded = connection.recv_bytes(maxlength=_MAX_HEADER_BYTES)
        value = json.loads(encoded.decode())
    except (EOFError, OSError) as exc:
        raise ReferenceProtocolError(
            "reference IPC control channel closed",
        ) from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReferenceProtocolError(
            "reference IPC header is not valid JSON",
        ) from exc
    if not isinstance(value, dict):
        raise ReferenceProtocolError(
            "reference IPC header must be a JSON object",
        )
    return value


def _validate_success_header(header: dict[str, Any]) -> None:
    if header.get("ok") is not True:
        raw_kind = header.get("failure_kind")
        try:
            failure_kind = ReferenceFailureKind(raw_kind)
        except (TypeError, ValueError) as exc:
            raise ReferenceProtocolError(
                "reference IPC failure category is invalid",
            ) from exc
        raise ReferenceExecutionError(
            str(header.get("error") or "reference failed"),
            kind=failure_kind,
        )
    if header.get("protocol") != SchemaVersion.REFERENCE_IPC:
        raise ReferenceProtocolError("reference IPC protocol version mismatch")


def _storage_span(tensor: torch.Tensor) -> int:
    if tensor.numel() == 0:
        return 0
    return 1 + sum(
        (size - 1) * stride
        for size, stride in zip(tensor.shape, tensor.stride(), strict=True)
        if size > 1
    )


def reference_values_storage_bytes(values: Iterable[Any]) -> int:
    """Return the physical tensor-storage bytes carried by reference IPC."""
    return sum(
        _storage_span(value) * value.element_size()
        for value in values
        if isinstance(value, torch.Tensor)
    )


def reference_case_storage_bytes(case: ReferenceCase) -> int:
    """Return the physical input/output bytes carried by one reference case."""
    return reference_values_storage_bytes(
        case.inputs,
    ) + reference_values_storage_bytes(
        case.outputs,
    )


def _encode_values(
    values: Iterable[Any],
    prefix: str,
) -> tuple[list[dict[str, Any]], dict[str, torch.Tensor]]:
    metadata: list[dict[str, Any]] = []
    tensors: dict[str, torch.Tensor] = {}
    for index, item in enumerate(values):
        value = item
        if not isinstance(value, torch.Tensor):
            if not isinstance(value, (bool, int, float)):
                raise ReferenceProtocolError(
                    f"unsupported reference IPC scalar: {type(value).__name__}",
                )
            metadata.append({"kind": "scalar", "value": value})
            continue
        if any(stride < 0 for stride in value.stride()):
            value = value.contiguous()
        key = f"{prefix}_{index}"
        span = _storage_span(value)
        if span:
            storage = value.as_strided(
                (span,),
                (1,),
                storage_offset=value.storage_offset(),
            )
            tensors[key] = storage.detach().cpu().contiguous()
        else:
            tensors[key] = torch.empty(0, dtype=value.dtype)
        metadata.append(
            {
                "kind": "tensor",
                "key": key,
                "shape": list(value.shape),
                "stride": list(value.stride()),
            },
        )
    return metadata, tensors


def _decode_values(
    metadata: list[dict[str, Any]],
    tensors: dict[str, torch.Tensor],
    *,
    device: str,
) -> list[Any]:
    values: list[Any] = []
    for item in metadata:
        kind = item.get("kind")
        if kind == "scalar":
            value = item.get("value")
            if not isinstance(value, (bool, int, float)):
                raise ReferenceProtocolError("invalid reference IPC scalar")
            values.append(value)
            continue
        if kind != "tensor":
            raise ReferenceProtocolError(
                f"unknown reference IPC value kind: {kind!r}",
            )
        key = item.get("key")
        shape = item.get("shape")
        stride = item.get("stride")
        if not isinstance(key, str) or key not in tensors:
            raise ReferenceProtocolError("reference IPC tensor key is missing")
        if not isinstance(shape, list) or not isinstance(stride, list):
            raise ReferenceProtocolError(
                "reference IPC tensor metadata is invalid",
            )
        storage = tensors[key].to(device=device)
        values.append(storage.as_strided(tuple(shape), tuple(stride), 0))
    return values


def _encoded_inputs_sha256(
    metadata: list[dict[str, Any]],
    tensors: dict[str, torch.Tensor],
) -> str:
    """Hash exactly the scalar metadata and tensor storage received over IPC."""
    input_tensors = {
        item["key"]: tensors[item["key"]]
        for item in metadata
        if item.get("kind") == "tensor"
        and isinstance(item.get("key"), str)
        and item["key"] in tensors
    }
    payload = save_safetensors_bytes(dict(sorted(input_tensors.items())))
    return stable_json_checksum(
        {
            "metadata": metadata,
            "tensor_payload_sha256": sha256_bytes(payload),
        },
    )


def send_case(
    connection: Connection,
    case: ReferenceCase,
    *,
    reference_latency_ms: float | None = None,
    timing_failure: str | None = None,
    access_patterns: Sequence[AccessPatternSummary] = (),
) -> None:
    """Send a reference case using JSON metadata and safetensors payload bytes."""
    input_metadata, input_tensors = _encode_values(case.inputs, "input")
    output_metadata, output_tensors = _encode_values(case.outputs, "output")
    payload = save_safetensors_bytes({**input_tensors, **output_tensors})
    if len(payload) > MAX_REFERENCE_PAYLOAD_BYTES:
        raise ReferenceProtocolError(
            "reference IPC tensor payload is too large",
        )
    send_json(
        connection,
        {
            "ok": True,
            "protocol": SchemaVersion.REFERENCE_IPC,
            "inputs": input_metadata,
            "outputs": output_metadata,
            "payload_bytes": len(payload),
            "reference_latency_ms": reference_latency_ms,
            "timing_failure": timing_failure,
            "access_patterns": [
                pattern.model_dump(mode="json") for pattern in access_patterns
            ],
        },
    )
    _send_bytes(connection, payload, channel="tensor")


def receive_case(connection: Connection, *, device: str) -> ReferenceTimingCase:
    """Receive and validate one trusted reference response."""
    header = receive_json(connection)
    _validate_success_header(header)
    expected_size = header.get("payload_bytes")
    if not isinstance(expected_size, int) or expected_size < 0:
        raise ReferenceProtocolError("reference IPC payload length is invalid")
    try:
        payload = connection.recv_bytes(maxlength=MAX_REFERENCE_PAYLOAD_BYTES)
    except (EOFError, OSError) as exc:
        message = (
            "reference IPC tensor payload is too large"
            if expected_size > MAX_REFERENCE_PAYLOAD_BYTES
            else "reference IPC tensor channel closed"
        )
        raise ReferenceProtocolError(message) from exc
    if len(payload) != expected_size:
        raise ReferenceProtocolError("reference IPC payload length mismatch")
    try:
        tensors = load_safetensors_bytes(payload)
    except Exception as exc:
        raise ReferenceProtocolError(
            "reference IPC tensor payload is invalid",
        ) from exc
    inputs_meta = header.get("inputs")
    outputs_meta = header.get("outputs")
    if not isinstance(inputs_meta, list) or not isinstance(outputs_meta, list):
        raise ReferenceProtocolError("reference IPC case metadata is missing")
    latency = header.get("reference_latency_ms")
    if latency is None:
        latency = 0.0
    if not isinstance(latency, (int, float)):
        raise ReferenceProtocolError("reference latency is not numeric")
    try:
        inputs = _decode_values(inputs_meta, tensors, device=device)
        outputs = _decode_values(outputs_meta, tensors, device=device)
        access_patterns = [
            AccessPatternSummary.model_validate(item)
            for item in header.get("access_patterns", [])
        ]
    except ReferenceProtocolError:
        raise
    except Exception as exc:
        raise ReferenceProtocolError(
            f"reference IPC tensor materialization failed on {device}",
        ) from exc
    return ReferenceTimingCase(
        inputs=inputs,
        outputs=outputs,
        reference_latency_ms=float(latency),
        input_sha256=_encoded_inputs_sha256(inputs_meta, tensors),
        access_patterns=access_patterns,
        timing_failure=(
            str(header["timing_failure"])
            if header.get("timing_failure")
            else None
        ),
    )


def send_failure(
    connection: Connection,
    message: str,
    *,
    kind: ReferenceFailureKind = ReferenceFailureKind.REFERENCE_EXECUTION,
) -> None:
    """Send a bounded trusted-reference failure without a tensor payload."""
    send_json(
        connection,
        {
            "ok": False,
            "protocol": SchemaVersion.REFERENCE_IPC,
            "error": message[:4096],
            "failure_kind": kind,
        },
    )


class ReferenceClient:
    """Candidate-side client for the isolated trusted reference service."""

    def __init__(
        self,
        reader: Connection,
        writer: Connection,
        *,
        token: str,
        device: str,
        worker_pid: int | None = None,
    ) -> None:
        """Initialize a client over authenticated IPC connections."""
        self._reader = reader
        self._writer = writer
        self._token = token
        self._device = device
        self._worker_pid = worker_pid
        self._closed = False

    def correctness_case(
        self,
        *,
        workload_uuid: str,
        row_index: int,
        round_index: int,
    ) -> ReferenceCase:
        """Request a fresh correctness case for one workload round."""
        response = self._request(
            "correctness",
            workload_uuid=workload_uuid,
            row_index=row_index,
            round_index=round_index,
        )
        return ReferenceCase(inputs=response.inputs, outputs=response.outputs)

    def timing_case(
        self,
        *,
        workload_uuid: str,
        row_index: int,
        round_index: int,
    ) -> ReferenceTimingCase:
        """Request a fresh timing case for one workload round."""
        return self._request(
            "timing",
            workload_uuid=workload_uuid,
            row_index=row_index,
            round_index=round_index,
        )

    def timing_iteration_case(
        self,
        *,
        workload_uuid: str,
        row_index: int,
        trial_index: int,
        iteration_index: int,
    ) -> ReferenceTimingCase:
        """Request the unique input and reference for one timed invocation."""
        return self._request(
            "timing_iteration",
            workload_uuid=workload_uuid,
            row_index=row_index,
            round_index=9,
            trial_index=trial_index,
            iteration_index=iteration_index,
        )

    def validate_timing_outputs(self, outputs: list[torch.Tensor]) -> None:
        """Ask the trusted worker to validate one timed result once."""
        if self._closed:
            raise ReferenceProtocolError("reference IPC client is closed")
        send_json(
            self._writer,
            {
                "protocol": SchemaVersion.REFERENCE_IPC,
                "token": self._token,
                "operation": "timing_validation",
            },
        )
        send_case(
            self._writer,
            ReferenceCase(inputs=[], outputs=outputs),
        )
        _validate_success_header(receive_json(self._reader))

    def _request(self, operation: str, **values: Any) -> ReferenceTimingCase:
        if self._closed:
            raise ReferenceProtocolError("reference IPC client is closed")
        send_json(
            self._writer,
            {
                "protocol": SchemaVersion.REFERENCE_IPC,
                "token": self._token,
                "operation": operation,
                **values,
            },
        )
        return receive_case(self._reader, device=self._device)

    def close(self) -> None:
        """Shut down the reference worker and close both IPC connections."""
        if self._closed:
            return
        try:
            with suppress(
                BrokenPipeError,
                EOFError,
                OSError,
                ReferenceProtocolError,
            ):
                send_json(
                    self._writer,
                    {
                        "protocol": SchemaVersion.REFERENCE_IPC,
                        "token": self._token,
                        "operation": "shutdown",
                    },
                )
                receive_json(self._reader)
        finally:
            self._closed = True
            self._writer.close()
            self._reader.close()
            if self._worker_pid is not None:
                with suppress(ChildProcessError):
                    os.waitpid(self._worker_pid, 0)


def connect_reference_worker(*, device: str) -> ReferenceClient:
    """Connect to the orchestrator-owned worker and scrub its credentials."""
    request_fd = os.environ.pop(REFERENCE_REQUEST_FD_ENV, None)
    response_fd = os.environ.pop(REFERENCE_RESPONSE_FD_ENV, None)
    token = os.environ.pop(REFERENCE_TOKEN_ENV, None)
    raw_pid = os.environ.pop(REFERENCE_PID_ENV, None)
    if request_fd is None or response_fd is None or not token:
        raise ReferenceProtocolError(
            "trusted reference worker is not configured",
        )
    writer = Connection(int(request_fd), readable=False, writable=True)
    reader = Connection(int(response_fd), readable=True, writable=False)
    worker_pid = int(raw_pid) if raw_pid else None
    return ReferenceClient(
        reader,
        writer,
        token=token,
        device=device,
        worker_pid=worker_pid,
    )


__all__ = [
    "REFERENCE_PID_ENV",
    "REFERENCE_REQUEST_FD_ENV",
    "REFERENCE_RESPONSE_FD_ENV",
    "REFERENCE_TOKEN_ENV",
    "TRUSTED_DEFINITION_FILE",
    "ReferenceCase",
    "ReferenceClient",
    "ReferenceExecutionError",
    "ReferenceFailureKind",
    "ReferenceProtocolError",
    "ReferenceTimingCase",
    "connect_reference_worker",
    "receive_json",
    "send_case",
    "send_failure",
    "send_json",
]
