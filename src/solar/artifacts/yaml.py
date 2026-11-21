# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""YAML serialization primitives for canonical SOLAR artifacts."""

from __future__ import annotations

import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import yaml


class NoAliasDumper(yaml.SafeDumper):
    """Disable YAML anchors so canonical artifacts remain reviewable."""

    def ignore_aliases(self, data: object) -> bool:
        """Disable aliases for every serialized value."""
        del data
        return True


def write_yaml_atomic(
    path: str | Path,
    payload: object,
    *,
    no_aliases: bool = True,
) -> Path:
    """Serialize YAML via a sibling temp file, fsync, and atomic replace."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    dumper = NoAliasDumper if no_aliases else yaml.SafeDumper
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            yaml.dump(
                payload,
                stream,
                Dumper=dumper,
                sort_keys=False,
                default_flow_style=False,
            )
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination


__all__ = ["NoAliasDumper", "write_yaml_atomic"]
