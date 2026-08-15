# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
"""Typed parsers for AMD SMI JSON used by benchmark isolation."""

from __future__ import annotations

import re
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, RootModel

type MetricJSON = (
    str | int | float | bool | None | list[MetricJSON] | dict[str, MetricJSON]
)


class AMDSMIPerformanceLevel(BaseModel):
    """Performance-level record for one GPU."""

    model_config = ConfigDict(extra="ignore")

    gpu: int | str
    perf_level: str


class AMDSMIPerformancePayload(BaseModel):
    """Top-level AMD SMI performance-level response."""

    model_config = ConfigDict(extra="ignore")

    gpu_data: list[AMDSMIPerformanceLevel]


class AMDSMIProcess(BaseModel):
    """Process record reported by AMD SMI."""

    model_config = ConfigDict(extra="ignore")

    pid: int | None = None
    name: str | None = None
    process_name: str | None = None


class AMDSMIGPUProcesses(BaseModel):
    """Processes associated with one GPU."""

    model_config = ConfigDict(extra="ignore")

    gpu: int | str
    process_list: list[AMDSMIProcess]


class AMDSMIProcessPayload(RootModel[list[AMDSMIGPUProcesses]]):
    """Root list of per-GPU process records."""


class AMDSMIGPUIdentity(BaseModel):
    """Minimal identity record for one GPU."""

    model_config = ConfigDict(extra="ignore")

    gpu: int | str
    bdf: str | None = None
    uuid: str | None = None


class AMDSMIListPayload(RootModel[list[AMDSMIGPUIdentity]]):
    """Root list returned by the AMD SMI list command."""


class AMDSMIMetricPayload(RootModel[dict[str, MetricJSON] | list[MetricJSON]]):
    """Version-tolerant root validated before metric field extraction."""


@dataclass(frozen=True, slots=True, kw_only=True)
class AMDSMIMetricObservation:
    """Normalized numeric telemetry for one GPU."""

    sclk_mhz: float | None
    mclk_mhz: float | None
    temperature_c: float | None
    power_cap_w: float | None
    power_draw_w: float | None
    power_profile: str | None


def parse_performance_levels(raw: str) -> tuple[str, ...]:
    """Return normalized, non-empty levels for every reported GPU."""
    payload = AMDSMIPerformancePayload.model_validate_json(raw)
    if not payload.gpu_data:
        raise ValueError("amd-smi returned no GPU performance-level data")
    levels = tuple(
        entry.perf_level.strip().upper() for entry in payload.gpu_data
    )
    if any(not level for level in levels):
        raise ValueError("amd-smi returned an empty GPU performance level")
    return levels


def parse_processes(raw: str) -> list[dict[str, int | str]]:
    """Return the stable process fields used by isolation snapshots."""
    payload = AMDSMIProcessPayload.model_validate_json(raw)
    processes: list[dict[str, int | str]] = []
    for gpu_entry in payload.root:
        for process in gpu_entry.process_list:
            if process.pid is None:
                continue
            processes.append(
                {
                    "pid": process.pid,
                    "device": str(gpu_entry.gpu),
                    "name": process.name or process.process_name or "unknown",
                },
            )
    return processes


def parse_gpu_count(raw: str) -> int:
    """Return the number of unique GPU identifiers in ``amd-smi list`` JSON."""
    payload = AMDSMIListPayload.model_validate_json(raw)
    return len({str(entry.gpu) for entry in payload.root})


def parse_gpu_identity(raw: str, gpu_index: int) -> AMDSMIGPUIdentity:
    """Return one exact GPU list identity with required UUID and BDF."""
    payload = AMDSMIListPayload.model_validate_json(raw)
    matches = [
        identity
        for identity in payload.root
        if str(identity.gpu) == str(gpu_index)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"amd-smi returned no unique identity for GPU {gpu_index}"
        )
    identity = matches[0]
    if not identity.uuid or not identity.bdf:
        raise ValueError("amd-smi GPU identity lacks UUID or BDF")
    return identity


def parse_gpu_metrics(raw: str, gpu_index: int) -> AMDSMIMetricObservation:
    """Extract stable telemetry across AMD SMI JSON naming variations."""
    payload = AMDSMIMetricPayload.model_validate_json(raw).root
    selected = _select_gpu_metric_payload(payload, gpu_index)
    flattened = _flatten_metric_payload(selected)
    return AMDSMIMetricObservation(
        sclk_mhz=_first_number(
            flattened,
            "clock_gfx_0_clk_value",
            "gfx_clock",
            "sclk",
            "clk_gfx",
        ),
        mclk_mhz=_first_number(
            flattened,
            "clock_mem_0_clk_value",
            "mem_clock",
            "mclk",
            "clk_mem",
        ),
        temperature_c=_first_number(
            flattened,
            "temperature_hotspot_value",
            "temperature_edge_value",
            "temperature_hotspot",
            "hotspot_temperature",
            "temperature_edge",
            "temperature",
        ),
        power_cap_w=_first_number(
            flattened,
            "power_cap",
            "socket_power_cap",
        ),
        power_draw_w=_first_number(
            flattened,
            "power_socket_power_value",
            "current_socket_power",
            "average_socket_power",
            "power_draw",
        ),
        power_profile=_first_text(
            flattened,
            "power_profile",
            "power_management",
        ),
    )


def _select_gpu_metric_payload(
    value: MetricJSON,
    gpu_index: int,
) -> MetricJSON:
    if isinstance(value, dict):
        gpu_data = value.get("gpu_data")
        if isinstance(gpu_data, list):
            return _select_gpu_metric_payload(gpu_data, gpu_index)
        return value
    if isinstance(value, list):
        for item in value:
            if not isinstance(item, dict):
                continue
            observed = item.get("gpu", item.get("gpu_id", item.get("device")))
            if str(observed) == str(gpu_index):
                return item
        if len(value) == 1:
            return value[0]
    raise ValueError(f"amd-smi metrics lack GPU {gpu_index}")


def _flatten_metric_payload(
    value: MetricJSON,
    *,
    prefix: str = "",
) -> dict[str, MetricJSON]:
    result: dict[str, MetricJSON] = {}
    if not isinstance(value, dict):
        return result
    for key, item in value.items():
        normalized = re.sub(r"[^a-z0-9]+", "_", str(key).lower()).strip("_")
        path = f"{prefix}_{normalized}".strip("_")
        if isinstance(item, dict):
            result.update(_flatten_metric_payload(item, prefix=path))
        else:
            result[path] = item
            result.setdefault(normalized, item)
    return result


def _first_number(
    values: dict[str, MetricJSON],
    *suffixes: str,
) -> float | None:
    for suffix in suffixes:
        for key, value in values.items():
            if key == suffix or key.endswith(f"_{suffix}"):
                match = re.search(r"[-+]?\d+(?:\.\d+)?", str(value))
                if match is not None:
                    return float(match.group())
    return None


def _first_text(
    values: dict[str, MetricJSON],
    *suffixes: str,
) -> str | None:
    for suffix in suffixes:
        for key, value in values.items():
            if key == suffix or key.endswith(f"_{suffix}"):
                text = str(value).strip()
                return text or None
    return None
