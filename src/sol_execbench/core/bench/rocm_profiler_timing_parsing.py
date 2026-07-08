# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""ROCm profiler timing parsing and live timing collection."""


import csv
import re
from pathlib import Path

from sol_execbench.core.bench.rocm_profiler_models import Rocprofv3TimingRow


def parse_rocprofv3_csv(content: str) -> tuple[Rocprofv3TimingRow, ...]:
    """Parse representative `rocprofv3` CSV content into timing rows."""
    reader = csv.DictReader(content.splitlines())
    rows: list[Rocprofv3TimingRow] = []
    for raw_row in reader:
        normalized = {_normalize_header(key): value for key, value in raw_row.items()}
        name = _first_value(normalized, "kernelname", "name", "function", "operation")
        domain = _first_value(normalized, "domain", "kind", "type", "category")
        duration_ns = _duration_ns(normalized)
        if name is None or domain is None or duration_ns is None:
            continue
        rows.append(
            Rocprofv3TimingRow(
                name=name,
                domain=domain,
                duration_ns=duration_ns,
                raw={key: value for key, value in raw_row.items() if key is not None},
            )
        )
    return tuple(rows)


def summarize_rocprofv3_csv(path: Path) -> tuple[int, float]:
    """Stream a `rocprofv3` CSV file into kernel row count and duration."""
    kernel_rows = 0
    duration_ns = 0.0
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw_row in reader:
            normalized = {
                _normalize_header(key): value for key, value in raw_row.items()
            }
            domain = _first_value(normalized, "domain", "kind", "type", "category")
            row_duration_ns = _duration_ns(normalized)
            if domain is None or row_duration_ns is None:
                continue
            if "kernel" not in _normalize_header(domain):
                continue
            kernel_rows += 1
            duration_ns += row_duration_ns
    return kernel_rows, duration_ns


def _normalize_header(header: str | None) -> str:
    return re.sub(r"[^a-z0-9]", "", (header or "").lower())


def _first_value(row: dict[str, str], *keys: str) -> str | None:
    for key in keys:
        value = row.get(key)
        if value:
            return value.strip()
    return None


def _duration_ns(row: dict[str, str]) -> float | None:
    for key in (
        "durationns",
        "durationnsec",
        "durationnanoseconds",
        "duration",
    ):
        value = row.get(key)
        if value:
            return float(value)

    start = _first_numeric(row, "starttimestamp", "startns", "begin")
    end = _first_numeric(row, "endtimestamp", "endns", "end")
    if start is not None and end is not None and end >= start:
        return end - start
    return None


def _first_numeric(row: dict[str, str], *keys: str) -> float | None:
    for key in keys:
        value = row.get(key)
        if value:
            return float(value)
    return None

normalize_header = _normalize_header
first_value = _first_value
duration_ns = _duration_ns
first_numeric = _first_numeric
