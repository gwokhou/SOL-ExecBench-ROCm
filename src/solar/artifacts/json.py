# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Atomic, canonical JSON artifacts for SOLAR-owned evidence."""

from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile


def write_json_artifact(path: Path, payload: object, *, max_bytes: int) -> None:
    """Publish within a byte budget, leaving existing evidence intact on failure."""
    if type(max_bytes) is not int or max_bytes <= 0:
        raise ValueError("JSON artifact byte budget must be a positive integer")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="wb", dir=path.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            remaining = max_bytes - 1
            encoder = json.JSONEncoder(
                sort_keys=True, indent=2, allow_nan=False
            )
            for chunk in encoder.iterencode(payload):
                if len(chunk) > remaining:
                    raise ValueError("JSON artifact exceeds the byte budget")
                encoded = chunk.encode("utf-8")
                if len(encoded) > remaining:
                    raise ValueError("JSON artifact exceeds the byte budget")
                stream.write(encoded)
                remaining -= len(encoded)
            stream.write(b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
