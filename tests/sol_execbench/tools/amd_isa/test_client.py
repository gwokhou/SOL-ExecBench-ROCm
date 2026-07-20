# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from io import StringIO
import json
from pathlib import Path
import signal
import subprocess
import threading

import pytest

from sol_execbench.tools.amd_isa import client as client_module
from sol_execbench.tools.amd_isa.client import AmdIsa, Decoder
from sol_execbench.tools.amd_isa.errors import IsaDecodeError, IsaProtocolError


class _Process:
    def __init__(self, *responses: str) -> None:
        self.stdin = StringIO()
        self.stdout = StringIO("".join(f"{response}\n" for response in responses))
        self.stderr = StringIO()
        self.pid = 1234
        self.returncode: int | None = None
        self.wait_results: list[BaseException | None] = []

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float) -> int:
        if self.wait_results:
            result = self.wait_results.pop(0)
            if result is not None:
                raise result
        self.returncode = 0
        return 0


def _response(request_id: int, **payload: object) -> str:
    return json.dumps({"id": request_id, **payload})


def _client(process: _Process) -> AmdIsa:
    client = AmdIsa.__new__(AmdIsa)
    client._process = process
    client._lock = threading.Lock()
    client._next_id = 1
    client._timeout_seconds = 1.0
    client.decoder = Decoder(client, "decoder")
    return client


@pytest.fixture(autouse=True)
def _make_stdout_readable(monkeypatch) -> None:
    monkeypatch.setattr(
        client_module.select, "select", lambda *_args: ([object()], [], [])
    )


def test_client_initializes_namespaces_and_merges_provenance(
    tmp_path: Path, monkeypatch
) -> None:
    process = _Process(
        _response(1, ok=True, result={"protocol": 1}),
        _response(2, ok=True, result={"spec": "loaded"}),
        _response(3, ok=True, result={"name": "s_add_u32"}),
    )
    monkeypatch.setattr(
        client_module.subprocess, "Popen", lambda *_args, **_kwargs: process
    )

    client = AmdIsa(
        tmp_path / "helper",
        tmp_path / "spec.xml",
        timeout_seconds=7,
        provenance={"release": "fixture"},
    )

    assert client.provenance == {"release": "fixture", "spec": "loaded"}
    assert client.explorer.get_instruction("s_add_u32") == {"name": "s_add_u32"}
    assert client._timeout_seconds == 7
    requests = [json.loads(line) for line in process.stdin.getvalue().splitlines()]
    assert [request["method"] for request in requests] == [
        "hello",
        "load",
        "explorer.get_instruction",
    ]


def test_call_rejects_stopped_helper() -> None:
    process = _Process()
    process.returncode = 1

    with pytest.raises(IsaProtocolError, match="not running"):
        _client(process)._call("hello", {})


def test_call_rejects_oversized_request(monkeypatch) -> None:
    monkeypatch.setattr(client_module, "_MAX_MESSAGE_BYTES", 32)

    with pytest.raises(ValueError, match="request exceeds"):
        _client(_Process())._call("decode", {"text": "x" * 100})


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ("", "closed its response stream"),
        ("not-json", "invalid JSON"),
        (_response(99, ok=True, result={}), "id does not match"),
    ],
)
def test_call_rejects_invalid_responses(raw: str, message: str) -> None:
    with pytest.raises(IsaProtocolError, match=message):
        _client(_Process(raw) if raw else _Process())._call("hello", {})


def test_call_rejects_oversized_response(monkeypatch) -> None:
    process = _Process(_response(1, ok=True, result="x" * 100))
    monkeypatch.setattr(client_module, "_MAX_MESSAGE_BYTES", 64)

    with pytest.raises(IsaProtocolError, match="response exceeds"):
        _client(process)._call("hello", {})


def test_call_maps_helper_error_to_decode_error() -> None:
    process = _Process(
        _response(1, ok=False, error={"code": "decode", "message": "bad word"})
    )

    with pytest.raises(IsaDecodeError, match="bad word"):
        _client(process)._call("decoder.decode_stream", {"words": [1]})


def test_call_reports_timeout_and_io_failure(monkeypatch) -> None:
    process = _Process()
    monkeypatch.setattr(client_module.select, "select", lambda *_args: ([], [], []))
    with pytest.raises(IsaProtocolError, match="timed out"):
        _client(process)._call("hello", {})

    class _BrokenInput(StringIO):
        def write(self, value: str) -> int:
            raise OSError("pipe closed")

    process = _Process()
    process.stdin = _BrokenInput()
    with pytest.raises(IsaProtocolError, match="communication failed"):
        _client(process)._call("hello", {})


def test_decoder_validates_integer_widths() -> None:
    decoder = _client(_Process()).decoder

    with pytest.raises(ValueError, match="unsigned 64-bit"):
        decoder.decode_machine_code(-1)
    with pytest.raises(ValueError, match="unsigned 64-bit"):
        decoder.decode_machine_code(1 << 64)
    with pytest.raises(ValueError, match="unsigned 32-bit"):
        decoder.decode_stream([1 << 32])


def test_close_escalates_when_helper_does_not_exit(monkeypatch) -> None:
    process = _Process(_response(1, ok=True, result=None))
    process.wait_results = [
        subprocess.TimeoutExpired("helper", 2),
        subprocess.TimeoutExpired("helper", 2),
        None,
    ]
    sent_signals: list[int] = []
    monkeypatch.setattr(
        client_module.os,
        "killpg",
        lambda _pid, signal_number: sent_signals.append(signal_number),
    )

    _client(process).close()

    assert sent_signals == [signal.SIGTERM, signal.SIGKILL]


def test_context_manager_closes_client(monkeypatch) -> None:
    client = _client(_Process())
    closed = False

    def close() -> None:
        nonlocal closed
        closed = True

    monkeypatch.setattr(client, "close", close)
    with client as entered:
        assert entered is client
    assert closed is True
