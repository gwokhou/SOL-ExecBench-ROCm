#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Stage raw authority inputs under self-contained publication-relative paths."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("closure_dir", type=Path)
    parser.add_argument("output_root", type=Path)
    args = parser.parse_args()
    source = args.closure_dir.resolve()
    output = args.output_root.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("publication source output must be new or empty")
    output.mkdir(parents=True, exist_ok=True)

    fixed = {
        "authority-suite-manifest-v4.json": "suite/authority-suite-manifest.json",
        "authority-inputs-v4-slice365.json": "authority.json",
        "baseline-v3-authority365-prepared/baseline-solution-portfolio.json": "candidate/solution.json",
        "baseline-v3-authority365-prepared/baseline-trace.jsonl": "candidate/trace.jsonl",
        "baseline-v3-authority365-rerun-prepared/baseline-trace.jsonl": "rerun/trace.jsonl",
        "gfx1200-full-suite-v3.model.json": "gfx1200-full-suite-v3.model.json",
        "calibration-primary-v3.json": "out/gfx1200-full-suite-closure/calibration-primary-v3.json",
        "calibration-verification-v3.json": "out/gfx1200-full-suite-closure/calibration-verification-v3.json",
        "hardware-profile-requirements.json": "out/gfx1200-full-suite-closure/hardware-profile-requirements.json",
    }
    for source_ref, destination_ref in fixed.items():
        _copy(source / source_ref, output / destination_ref)
    _copy(
        source / "baseline-v3-authority365-prepared/baseline-trace.jsonl",
        output / "candidate/timing-evidence.jsonl",
    )

    authority = _load(source / "authority-inputs-v4-slice365.json")
    rows = authority.get("workloads")
    if not isinstance(rows, list):
        raise ValueError("authority input must contain workloads")
    fusion_destinations: set[Path] = set()
    bound_count = 0
    for row in rows:
        if not isinstance(row, dict) or row.get("official_blockers"):
            raise ValueError("publication authority must contain only unblocked rows")
        bound_ref = row.get("bound_ref")
        if not isinstance(bound_ref, str):
            raise ValueError("publication authority row is missing bound_ref")
        bound_source = source / bound_ref
        bound_destination = output / bound_ref
        _copy(bound_source, bound_destination)
        bound = _load(bound_source)
        fusion_ref = bound.get("fusion_validation_ref")
        if not isinstance(fusion_ref, str):
            raise ValueError(f"{bound_ref} has no fusion validation reference")
        fusion_source = source / fusion_ref
        fusion_destination = bound_destination.parent / fusion_ref
        if fusion_destination not in fusion_destinations:
            _copy(fusion_source, fusion_destination)
            fusion_destinations.add(fusion_destination)
        bound_count += 1

    print(
        json.dumps(
            {
                "bounds": bound_count,
                "fusion_artifacts": len(fusion_destinations),
                "output_root": str(output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
