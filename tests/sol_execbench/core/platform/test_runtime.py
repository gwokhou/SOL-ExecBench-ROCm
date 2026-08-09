from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from sol_execbench.core.platform.runtime import (
    FALLBACK_CACHE_CLEAR_BYTES,
    collect_pcie_topology,
    derive_cache_clear_policy,
    detect_rocm_device,
    discover_rocm_root,
    hardware_from_device,
    pin_cuda_device,
    resolve_rocm_tool,
    resolve_rocm_tool_command,
    resolve_tool_path,
    rocm_search_roots,
)


def _pcie_link(
    path: Path,
    *,
    current_width: int,
    max_width: int,
) -> None:
    path.mkdir(parents=True)
    values = {
        "current_link_speed": "32.0 GT/s PCIe\n",
        "max_link_speed": "32.0 GT/s PCIe\n",
        "current_link_width": f"{current_width}\n",
        "max_link_width": f"{max_width}\n",
    }
    for name, value in values.items():
        (path / name).write_text(value, encoding="utf-8")


def test_collect_pcie_topology_binds_complete_root_to_endpoint_path(
    tmp_path: Path,
) -> None:
    sys_devices = tmp_path / "sys/devices"
    root = sys_devices / "pci0000:00/0000:00:01.1"
    upstream = root / "0000:01:00.0"
    downstream = upstream / "0000:02:00.0"
    endpoint = downstream / "0000:03:00.0"
    _pcie_link(root, current_width=8, max_width=8)
    _pcie_link(upstream, current_width=8, max_width=16)
    _pcie_link(downstream, current_width=16, max_width=16)
    _pcie_link(endpoint, current_width=16, max_width=16)
    pci_devices = tmp_path / "sys/bus/pci/devices"
    pci_devices.mkdir(parents=True)
    (pci_devices / endpoint.name).symlink_to(endpoint, target_is_directory=True)

    topology = collect_pcie_topology(
        endpoint.name,
        pci_devices_root=pci_devices,
        sys_devices_root=sys_devices,
    )

    assert [link.bdf for link in topology.links] == [
        "0000:00:01.1",
        "0000:01:00.0",
        "0000:02:00.0",
        "0000:03:00.0",
    ]
    assert topology.bottleneck_bdf == "0000:00:01.1"
    assert topology.effective_speed_gtps == 32.0
    assert topology.effective_width == 8


def test_collect_pcie_topology_rejects_incomplete_parent_link(
    tmp_path: Path,
) -> None:
    sys_devices = tmp_path / "sys/devices"
    root = sys_devices / "pci0000:00/0000:00:01.1"
    endpoint = root / "0000:03:00.0"
    root.mkdir(parents=True)
    _pcie_link(endpoint, current_width=16, max_width=16)
    pci_devices = tmp_path / "sys/bus/pci/devices"
    pci_devices.mkdir(parents=True)
    (pci_devices / endpoint.name).symlink_to(endpoint, target_is_directory=True)

    with pytest.raises(FileNotFoundError):
        collect_pcie_topology(
            endpoint.name,
            pci_devices_root=pci_devices,
            sys_devices_root=sys_devices,
        )


def test_cache_clear_policy_uses_twice_detected_l2() -> None:
    policy = derive_cache_clear_policy(4 * 1024**2)

    assert policy.detected_l2_bytes == 4 * 1024**2
    assert policy.clear_buffer_bytes == 8 * 1024**2
    assert policy.source == "torch_device_properties"
    assert policy.fallback_reason is None


def test_cache_clear_policy_falls_back_when_l2_is_unavailable() -> None:
    policy = derive_cache_clear_policy(None)

    assert policy.detected_l2_bytes is None
    assert policy.clear_buffer_bytes == FALLBACK_CACHE_CLEAR_BYTES
    assert policy.source == "fallback_default"
    assert policy.fallback_reason == "l2_cache_size_unavailable"


def test_detect_rocm_device_reads_exact_arch_and_l2_properties() -> None:
    properties = SimpleNamespace(
        name="AMD test GPU",
        gcnArchName="gfx1150:xnack-",
        total_memory=32 * 1024**3,
        L2_cache_size=16 * 1024**2,
    )
    rocm_device_api = SimpleNamespace(
        is_available=lambda: True,
        current_device=lambda: 0,
        device_count=lambda: 2,
        get_device_properties=lambda index: properties,
    )
    fake_torch = SimpleNamespace(
        __version__="2.9-test",
        version=SimpleNamespace(hip="7.2-test"),
        **{"cuda": rocm_device_api},  # noqa: PIE804 -- Audited namespace form
        device=lambda value: SimpleNamespace(type="cuda", index=int(value[-1])),
    )

    result = detect_rocm_device("cuda:1", torch_module=fake_torch)

    assert result.device == "cuda:1"
    assert result.index == 1
    assert result.gfx_target == "gfx1150"
    assert result.l2_cache_bytes == 16 * 1024**2


def test_hardware_from_device_supports_cpu() -> None:
    assert hardware_from_device("cpu")


def test_discover_rocm_root_prefers_configured_directory(tmp_path) -> None:
    configured = tmp_path / "configured-rocm"
    configured.mkdir()

    root = discover_rocm_root(
        environ={"ROCM_PATH": str(configured)},
        which=lambda _: None,
    )

    assert root == configured.resolve()


def test_discover_rocm_root_follows_hipcc_symlink(tmp_path) -> None:
    root = tmp_path / "rocm-7.2"
    hipcc = root / "bin" / "hipcc"
    hipcc.parent.mkdir(parents=True)
    hipcc.write_text("#!/bin/sh\n", encoding="utf-8")
    wrapper = tmp_path / "bin" / "hipcc"
    wrapper.parent.mkdir()
    wrapper.symlink_to(hipcc)

    assert resolve_tool_path("hipcc", which=lambda _: str(wrapper)) == hipcc
    assert discover_rocm_root(environ={}, which=lambda _: str(wrapper)) == root


def test_resolve_rocm_tool_falls_back_to_configured_root(tmp_path) -> None:
    root = tmp_path / "rocm"
    rocminfo = root / "bin" / "rocminfo"
    rocminfo.parent.mkdir(parents=True)
    rocminfo.write_text("tool", encoding="utf-8")

    assert (
        resolve_rocm_tool(
            "rocminfo",
            environ={"ROCM_PATH": str(root)},
            which=lambda _: None,
        )
        == rocminfo
    )


def test_resolve_rocm_tool_command_preserves_path_symlink(tmp_path) -> None:
    target = tmp_path / "rocm-7.2/bin/amd-smi"
    target.parent.mkdir(parents=True)
    target.write_text("tool", encoding="utf-8")
    wrapper = tmp_path / "rocm/bin/amd-smi"
    wrapper.parent.mkdir(parents=True)
    wrapper.symlink_to(target)

    assert resolve_rocm_tool_command(
        "amd-smi",
        which=lambda _: str(wrapper),
    ) == str(
        wrapper,
    )


def test_resolve_rocm_tool_command_falls_back_to_configured_root(
    tmp_path,
) -> None:
    root = tmp_path / "rocm"
    amd_smi = root / "bin/amd-smi"
    amd_smi.parent.mkdir(parents=True)
    amd_smi.write_text("tool", encoding="utf-8")

    assert resolve_rocm_tool_command(
        "amd-smi",
        environ={"ROCM_PATH": str(root)},
        which=lambda _: None,
    ) == str(amd_smi)


def test_rocm_search_roots_puts_discovered_root_first(tmp_path) -> None:
    root = tmp_path / "custom-rocm"
    root.mkdir()

    roots = rocm_search_roots(
        environ={"ROCM_PATH": str(root)},
        which=lambda _: None,
    )

    assert roots[0] == root.resolve()


def test_pin_cuda_device_noop_for_cpu_target() -> None:
    sink: list[int] = []
    fake = SimpleNamespace(
        cuda=SimpleNamespace(set_device=lambda index: sink.append(index)),
        device=lambda value: SimpleNamespace(type="cpu", index=None),
    )

    pin_cuda_device("cpu", torch_module=fake)

    assert sink == []


def test_pin_cuda_device_noop_when_cuda_unavailable() -> None:
    sink: list[int] = []
    fake = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: False,
            set_device=lambda index: sink.append(index),
        ),
        device=lambda value: SimpleNamespace(type="cuda", index=0),
    )

    pin_cuda_device("cuda:0", torch_module=fake)

    assert sink == []


def test_pin_cuda_device_pins_explicit_index() -> None:
    sink: list[int] = []
    fake = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            current_device=lambda: 0,
            set_device=lambda index: sink.append(index),
        ),
        device=lambda value: SimpleNamespace(
            type="cuda",
            index=int(value.split(":")[1]),
        ),
    )

    pin_cuda_device("cuda:1", torch_module=fake)

    assert sink == [1]


def test_pin_cuda_device_pins_current_device_when_index_unspecified() -> None:
    sink: list[int] = []
    fake = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            current_device=lambda: 2,
            set_device=lambda index: sink.append(index),
        ),
        device=lambda value: SimpleNamespace(type="cuda", index=None),
    )

    pin_cuda_device("cuda", torch_module=fake)

    assert sink == [2]
