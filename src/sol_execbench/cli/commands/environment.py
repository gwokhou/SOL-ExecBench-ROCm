# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Environment snapshot sidecar helpers for the SOL-ExecBench CLI."""

from __future__ import annotations

import os
from pathlib import Path

from rich.console import Console

from ...core.environment import collect_environment_snapshot
from ...core.runtime_evidence import write_json_payload

console = Console(stderr=True)

ENV_SNAPSHOT_ENABLE_ENV = "SOLEXECBENCH_ENV_SNAPSHOT"
ENV_SNAPSHOT_PATH_ENV = "SOLEXECBENCH_ENV_SNAPSHOT_PATH"


def _environment_snapshot_sidecar_path(output_file: Path | None) -> Path | None:
    """Return the optional environment snapshot sidecar path for this run."""

    explicit = os.environ.get(ENV_SNAPSHOT_PATH_ENV)
    if explicit:
        return Path(explicit)
    if os.environ.get(ENV_SNAPSHOT_ENABLE_ENV) == "1" and output_file is not None:
        return output_file.with_name(f"{output_file.name}.environment.json")
    return None


def _write_environment_snapshot_sidecar(
    output_file: Path | None,
    *,
    collector=collect_environment_snapshot,
) -> Path | None:
    """Write optional environment snapshot sidecar metadata.

    Snapshot evidence is diagnostic only. Collection or serialization failures
    are reported to stderr and never change benchmark correctness status.
    """

    sidecar_path = _environment_snapshot_sidecar_path(output_file)
    if sidecar_path is None:
        if os.environ.get(ENV_SNAPSHOT_ENABLE_ENV) == "1":
            console.print(
                "[yellow]Environment snapshot requested but no output path is available; "
                f"set {ENV_SNAPSHOT_PATH_ENV} or use --output.[/yellow]"
            )
        return None

    try:
        snapshot = collector()
        write_json_payload(sidecar_path, snapshot)
        console.print(f"[green]Saved environment snapshot to {sidecar_path}[/green]")
        return sidecar_path
    except Exception as exc:
        console.print(f"[yellow]Environment snapshot skipped: {exc}[/yellow]")
        return None
