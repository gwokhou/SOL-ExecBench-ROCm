# SPDX-License-Identifier: Apache-2.0
"""Identities of the packaged scope executor and its generated-driver API."""

from importlib.resources import files
from pathlib import Path

from sol_execbench.core.integrity.checksums import sha256_bytes

SCOPE_RUNTIME_SOURCES = (
    "driver/scope_replay.py",
    "driver/scope_environment.py",
    "driver/eval_runtime_api.py",
    "driver/eval_runtime_api_exports.py",
)


def scope_runtime_digests(archive: Path | None = None) -> dict[str, str]:
    """Optionally retain the exact executor sources before candidate launch."""
    result = {}
    package = files("sol_execbench")
    for name in SCOPE_RUNTIME_SOURCES:
        payload = package.joinpath(name).read_bytes()
        result[name] = sha256_bytes(payload)
        if archive is not None:
            destination = archive / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(payload)
    return result
