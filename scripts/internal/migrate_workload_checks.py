#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""One-shot migration from workload tolerance to named output checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _migrate_problem(workload_path: Path) -> bool:
    definition_path = workload_path.with_name("definition.json")
    if not definition_path.is_file():
        return False
    definition = json.loads(definition_path.read_text(encoding="utf-8"))
    output_names = list((definition.get("outputs") or {}).keys())
    changed = False
    records: list[dict[str, object]] = []
    for line in workload_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        tolerance = record.pop("tolerance", None)
        if "checks" not in record:
            record["checks"] = [
                {"type": "numeric", "output": name, **(tolerance or {})}
                for name in output_names
            ]
            changed = True
        records.append(record)
    if changed:
        text = "".join(
            json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
            for record in records
        )
        workload_path.write_text(text, encoding="utf-8")
    return changed


def _migrate_calibration(path: Path, problems_root: Path) -> bool:
    if not path.is_file():
        return False
    data = json.loads(path.read_text(encoding="utf-8"))
    if int(data.get("schema_version", 0)) != 1:
        return False
    for record in data.get("records") or []:
        tolerance = record.pop("tolerance", None)
        if tolerance is None:
            continue
        problem = problems_root / str(record["problem_path"])
        definition = json.loads(
            (problem / "definition.json").read_text(encoding="utf-8"),
        )
        record["checks"] = [
            {"type": "numeric", "output": name, **tolerance}
            for name in definition["outputs"]
        ]
        observed_atol = record.pop("observed_max_atol", 0.0)
        observed_rtol = record.pop("observed_max_rtol", 0.0)
        record["observed_outputs"] = {
            name: [observed_atol, observed_rtol]
            for name in definition["outputs"]
        }
    data["schema_version"] = 2
    path.write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return True


def main() -> None:
    """Migrate every problem below the requested roots."""
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--problems-root", type=Path)
    args = parser.parse_args()
    changed = 0
    for root in args.roots:
        paths = (
            [root]
            if root.name == "workload.jsonl"
            else root.rglob("workload.jsonl")
        )
        changed += sum(_migrate_problem(path) for path in paths)
    if args.calibration is not None:
        if args.problems_root is None:
            parser.error("--calibration requires --problems-root")
        changed += int(
            _migrate_calibration(args.calibration, args.problems_root)
        )
    print(f"migrated {changed} workload files")


if __name__ == "__main__":
    main()
