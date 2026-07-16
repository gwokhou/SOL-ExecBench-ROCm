# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Stable Python facade for the persistent AMD ISA JSON helper."""

from __future__ import annotations

import json
import os
from pathlib import Path
import select
import signal
import subprocess
import threading
from typing import Any, Mapping

from sol_execbench.tools.amd_isa.errors import IsaDecodeError, IsaProtocolError
from sol_execbench.tools.amd_isa.helper import ensure_helper
from sol_execbench.tools.amd_isa.repository import IsaSpecRepository


_MAX_MESSAGE_BYTES = 64 * 1024 * 1024


class _Endpoint:
    def __init__(self, client: AmdIsa, prefix: str) -> None:
        self._client = client
        self._prefix = prefix

    def _call(self, operation: str, **params: object) -> Any:
        return self._client._call(f"{self._prefix}.{operation}", params)


class Decoder(_Endpoint):
    """Decoder operations backed by ``amdisa::IsaDecoder``."""

    def get_instruction(self, name: str) -> Mapping[str, Any]:
        return self._call("get_instruction", name=name)

    def decode_machine_code(self, machine_code: int) -> list[Mapping[str, Any]]:
        if not 0 <= machine_code <= 0xFFFFFFFFFFFFFFFF:
            raise ValueError("machine_code must be an unsigned 64-bit integer")
        return self._call("get_instruction", machine_code=machine_code)

    def decode_stream(self, words: list[int]) -> list[list[Mapping[str, Any]]]:
        if any(not 0 <= word <= 0xFFFFFFFF for word in words):
            raise ValueError("instruction words must be unsigned 32-bit integers")
        return self._call("decode_stream", words=words)

    def decode_disassembly(
        self, text: str, *, resolve_direct_branch_targets: bool = False
    ) -> list[list[Mapping[str, Any]]]:
        return self._call(
            "decode_disassembly",
            text=text,
            resolve_direct_branch_targets=resolve_direct_branch_targets,
        )


class Explorer(_Endpoint):
    """Explorer operations backed by ``amdisa::explorer::Spec``."""

    def architecture(self) -> Mapping[str, Any]:
        return self._call("architecture")

    def list_instructions(self) -> list[Mapping[str, Any]]:
        return self._call("list_instructions")

    def get_instruction(self, name: str) -> Mapping[str, Any]:
        return self._call("get_instruction", name=name)

    def list_data_formats(self) -> list[Mapping[str, Any]]:
        return self._call("list_data_formats")

    def get_data_format(self, name: str) -> Mapping[str, Any]:
        return self._call("get_data_format", name=name)

    def list_operand_types(self) -> list[Mapping[str, Any]]:
        return self._call("list_operand_types")

    def get_operand_type(self, name: str) -> Mapping[str, Any]:
        return self._call("get_operand_type", name=name)

    def list_functional_groups(self) -> list[Mapping[str, Any]]:
        return self._call("list_functional_groups")

    def get_functional_group(self, name: str) -> Mapping[str, Any]:
        return self._call("get_functional_group", name=name)


class AmdIsa:
    """A loaded ISA specification with decoder and explorer namespaces."""

    def __init__(
        self,
        helper: Path,
        spec_path: Path,
        *,
        timeout_seconds: float = 120.0,
        provenance: Mapping[str, Any] | None = None,
    ) -> None:
        self._process = subprocess.Popen(
            [str(helper)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        self._lock = threading.Lock()
        self._next_id = 1
        self._timeout_seconds = 30.0
        self.decoder = Decoder(self, "decoder")
        self.explorer = Explorer(self, "explorer")
        try:
            self._call("hello", {})
            loaded = self._call("load", {"spec_path": str(spec_path)})
        except BaseException:
            self._terminate_process_group(signal.SIGTERM)
            try:
                self._process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._terminate_process_group(signal.SIGKILL)
                self._process.wait(timeout=2)
            raise
        self._provenance = {**dict(provenance or {}), **dict(loaded)}
        self._timeout_seconds = timeout_seconds

    @property
    def provenance(self) -> Mapping[str, Any]:
        return self._provenance

    def _call(self, method: str, params: Mapping[str, object]) -> Any:
        with self._lock:
            if (
                self._process.poll() is not None
                or self._process.stdin is None
                or self._process.stdout is None
            ):
                raise IsaProtocolError("AMD ISA helper is not running")
            request_id = self._next_id
            self._next_id += 1
            request = {
                "protocol_version": 1,
                "id": request_id,
                "method": method,
                "params": dict(params),
            }
            encoded = json.dumps(request, separators=(",", ":"))
            if len(encoded.encode("utf-8")) > _MAX_MESSAGE_BYTES:
                raise ValueError("AMD ISA request exceeds protocol size limit")
            try:
                self._process.stdin.write(encoded + "\n")
                self._process.stdin.flush()
                readable, _, _ = select.select(
                    [self._process.stdout], [], [], self._timeout_seconds
                )
                if not readable:
                    raise IsaProtocolError("AMD ISA helper response timed out")
                raw = self._process.stdout.readline()
            except OSError as exc:
                raise IsaProtocolError("AMD ISA helper communication failed") from exc
            if not raw:
                raise IsaProtocolError("AMD ISA helper closed its response stream")
            if len(raw.encode("utf-8")) > _MAX_MESSAGE_BYTES:
                raise IsaProtocolError(
                    "AMD ISA helper response exceeds protocol size limit"
                )
            try:
                response = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise IsaProtocolError("AMD ISA helper returned invalid JSON") from exc
            if response.get("id") != request_id:
                raise IsaProtocolError(
                    "AMD ISA helper response id does not match request"
                )
            if response.get("ok") is True:
                return response.get("result")
            error = response.get("error", {})
            raise IsaDecodeError(str(error.get("message", "AMD ISA operation failed")))

    def close(self) -> None:
        if self._process.poll() is None:
            try:
                self._call("shutdown", {})
            except (IsaDecodeError, IsaProtocolError):
                pass
            try:
                self._process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._terminate_process_group(signal.SIGTERM)
                try:
                    self._process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self._terminate_process_group(signal.SIGKILL)
                    self._process.wait(timeout=2)

    def _terminate_process_group(self, signal_number: int) -> None:
        try:
            os.killpg(self._process.pid, signal_number)
        except ProcessLookupError:
            pass

    def __enter__(self) -> AmdIsa:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def open_isa(
    architecture: str,
    *,
    repository: IsaSpecRepository | None = None,
    allow_download: bool = True,
    timeout_seconds: float = 120.0,
) -> AmdIsa:
    """Open one architecture's specification, building/downloading on demand."""
    repository = repository or IsaSpecRepository()
    descriptor = repository.resolve(architecture, allow_download=allow_download)
    return AmdIsa(
        ensure_helper(repository.cache_root),
        descriptor.path,
        timeout_seconds=timeout_seconds,
        provenance={
            "architecture_target": descriptor.architecture,
            "family": descriptor.family,
            "release": descriptor.release,
            "spec_sha256": descriptor.sha256,
        },
    )
