from __future__ import annotations

import pytest
from pydantic import ValidationError

from sol_execbench.core.platform.amd_smi import (
    parse_gpu_count,
    parse_gpu_identity,
    parse_gpu_metrics,
    parse_performance_levels,
    parse_processes,
)


def test_parse_performance_levels_requires_nonempty_gpu_data() -> None:
    assert parse_performance_levels(
        '{"gpu_data": ['
        '{"gpu": 0, "perf_level": "AMDSMI_DEV_PERF_LEVEL_AUTO"},'
        '{"gpu": 1, "perf_level": "AMDSMI_DEV_PERF_LEVEL_STABLE_PEAK"}'
        "]}",
    ) == (
        "AMDSMI_DEV_PERF_LEVEL_AUTO",
        "AMDSMI_DEV_PERF_LEVEL_STABLE_PEAK",
    )

    with pytest.raises(ValueError, match="no GPU"):
        parse_performance_levels('{"gpu_data": []}')


def test_parse_performance_levels_rejects_invalid_shape() -> None:
    with pytest.raises(ValidationError):
        parse_performance_levels("[]")


def test_parse_processes_ignores_no_process_marker() -> None:
    assert (
        parse_processes(
            '[{"gpu": 0, "process_list": '
            '[{"process_info": "No running processes detected"}]}]',
        )
        == []
    )


def test_parse_processes_normalizes_stable_fields() -> None:
    assert parse_processes(
        '[{"gpu": 0, "process_list": ['
        '{"pid": 42, "process_name": "worker", "memory_usage": 10}'
        "]}]",
    ) == [{"pid": 42, "device": "0", "name": "worker"}]


def test_parse_gpu_count_counts_unique_ids() -> None:
    assert parse_gpu_count('[{"gpu": 0}, {"gpu": 1}, {"gpu": 1}]') == 2


def test_parse_gpu_identity_requires_exact_uuid_and_bdf() -> None:
    identity = parse_gpu_identity(
        '[{"gpu":0,"bdf":"0000:03:00.0","uuid":"gpu-uuid"}]',
        0,
    )

    assert (identity.uuid, identity.bdf) == ("gpu-uuid", "0000:03:00.0")


def test_parse_gpu_metrics_normalizes_nested_units() -> None:
    metrics = parse_gpu_metrics(
        '{"gpu_data":[{"gpu":0,"clock":{"gfx_clock":"3200 MHz",'
        '"mem_clock":"1250 MHz"},"temperature":{"hotspot_temperature":'
        '"52.5 C"},"power":{"current_socket_power":"101 W",'
        '"power_cap":"182 W","power_profile":"COMPUTE"}}]}',
        0,
    )

    assert metrics.sclk_mhz == 3200.0
    assert metrics.mclk_mhz == 1250.0
    assert metrics.temperature_c == 52.5
    assert metrics.power_draw_w == 101.0
    assert metrics.power_cap_w == 182.0
    assert metrics.power_profile == "COMPUTE"
