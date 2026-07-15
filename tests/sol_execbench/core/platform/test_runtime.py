from __future__ import annotations

from sol_execbench.core.platform.runtime import (
    discover_rocm_root,
    hardware_from_device,
    resolve_rocm_tool,
    resolve_tool_path,
    rocm_search_roots,
)


def test_hardware_from_device_supports_cpu() -> None:
    assert hardware_from_device("cpu")


def test_discover_rocm_root_prefers_configured_directory(tmp_path) -> None:
    configured = tmp_path / "configured-rocm"
    configured.mkdir()

    root = discover_rocm_root(
        environ={"ROCM_PATH": str(configured)}, which=lambda _: None
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
            "rocminfo", environ={"ROCM_PATH": str(root)}, which=lambda _: None
        )
        == rocminfo
    )


def test_rocm_search_roots_puts_discovered_root_first(tmp_path) -> None:
    root = tmp_path / "custom-rocm"
    root.mkdir()

    roots = rocm_search_roots(environ={"ROCM_PATH": str(root)}, which=lambda _: None)

    assert roots[0] == root.resolve()
