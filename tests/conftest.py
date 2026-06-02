# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
from collections.abc import Callable
from pathlib import Path

import pytest

PathExists = Callable[[Path], bool]

_ROCM_DEVICE_NODES = (Path("/dev/kfd"), Path("/dev/dri"))


def _missing_rocm_device_nodes(path_exists: PathExists | None = None) -> tuple[str, ...]:
    """Return missing ROCm device nodes for Linux hardware test collection."""
    if sys.platform != "linux":
        return ()
    exists = path_exists or Path.exists
    return tuple(str(path) for path in _ROCM_DEVICE_NODES if not exists(path))


def _rocm_device_node_skip_reason(missing_nodes: tuple[str, ...]) -> str:
    return (
        "ROCm device nodes unavailable in current execution environment: "
        + ", ".join(missing_nodes)
        + ". If this is a Codex or container sandbox, run GPU checks with ROCm "
        "device passthrough."
    )


def _rocm_gpu_info(path_exists: PathExists | None = None) -> tuple[bool, str, str]:
    """Return ROCm GPU availability, AMD gfx architecture, and skip reason."""
    missing_nodes = _missing_rocm_device_nodes(path_exists)
    if missing_nodes:
        return False, "", _rocm_device_node_skip_reason(missing_nodes)

    try:
        import torch

        if getattr(torch.version, "hip", None) is None:
            return False, "", "PyTorch is not a ROCm build"
        if not torch.cuda.is_available():
            return False, "", "ROCm GPU unavailable through PyTorch"
        props = torch.cuda.get_device_properties(0)
        arch = getattr(props, "gcnArchName", "") or getattr(props, "gfx_arch_name", "")
        return True, str(arch).split(":", maxsplit=1)[0], ""
    except ImportError as exc:
        return False, "", f"PyTorch import failed: {exc}"
    except (RuntimeError, AttributeError) as exc:
        return False, "", f"PyTorch ROCm probe failed: {exc}"


def _has_rocm_dev_headers() -> bool:
    """Return whether ROCm native extension development headers are installed."""
    rocm_root = Path("/opt/rocm")
    return (rocm_root / "include/hip/hip_runtime_api.h").exists()


def _has_ck_headers() -> bool:
    """Return whether Composable Kernel headers are installed."""
    return Path("/opt/rocm/include/ck/ck.hpp").exists()


def _has_rocwmma_headers() -> bool:
    """Return whether rocWMMA headers are installed."""
    return Path("/opt/rocm/include/rocwmma/rocwmma.hpp").exists()


def _is_rdna4(gfx_arch: str) -> bool:
    return gfx_arch.startswith("gfx12")


def _is_cdna3(gfx_arch: str) -> bool:
    return gfx_arch.startswith("gfx94")


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "timing_serial: GPU timing tests (skipped by default; run with: pytest tests -m timing_serial -n 0)",
    )
    config.addinivalue_line(
        "markers",
        "requires_rocm: test requires a ROCm GPU visible through PyTorch",
    )
    config.addinivalue_line(
        "markers",
        "requires_rocm_dev: test requires ROCm native extension development headers",
    )
    config.addinivalue_line(
        "markers",
        "requires_ck: test requires Composable Kernel headers",
    )
    config.addinivalue_line(
        "markers",
        "requires_rocwmma: test requires rocWMMA headers",
    )
    config.addinivalue_line(
        "markers",
        "requires_rdna4: test requires an AMD RDNA 4 GPU, such as gfx1200",
    )
    config.addinivalue_line(
        "markers",
        "requires_cdna3: test requires an AMD CDNA 3 GPU, such as gfx942",
    )
    config.addinivalue_line(
        "markers",
        "requires_cutile: legacy NVIDIA cuTile marker; skipped in this ROCm-only port",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Skip tests based on hardware availability.

    Also skips timing_serial tests unless explicitly selected with -m.
    """
    rocm_available, gfx_arch, rocm_skip_reason = _rocm_gpu_info()
    rocm_dev_available = _has_rocm_dev_headers()
    ck_available = _has_ck_headers()
    rocwmma_available = _has_rocwmma_headers()
    detected = gfx_arch or "unavailable"
    supported_arch = _is_rdna4(gfx_arch) or _is_cdna3(gfx_arch)
    skip_timing = pytest.mark.skip(
        reason="timing_serial tests skipped by default; run with: pytest tests -m timing_serial -n 0"
    )
    skip_no_rocm = pytest.mark.skip(reason=rocm_skip_reason)
    skip_no_rocm_dev = pytest.mark.skip(reason="ROCm HIP development headers unavailable")
    skip_no_ck = pytest.mark.skip(reason="Composable Kernel headers unavailable")
    skip_no_rocwmma = pytest.mark.skip(reason="rocWMMA headers unavailable")
    skip_rdna4 = pytest.mark.skip(
        reason=f"requires AMD RDNA 4 ROCm GPU (detected {detected})"
    )
    skip_cdna3 = pytest.mark.skip(
        reason=f"requires AMD CDNA 3 ROCm GPU (detected {detected})"
    )
    skip_unsupported = pytest.mark.skip(
        reason=f"unsupported AMD GPU architecture for ROCm test (detected {detected})"
    )
    skip_legacy_cutile = pytest.mark.skip(
        reason="legacy cuTile tests are NVIDIA-only and unavailable in this ROCm-only port"
    )

    # If the user passed -m that includes timing_serial, don't auto-skip them.
    markexpr = config.getoption("-m", default="")
    timing_selected = "timing_serial" in markexpr

    for item in items:
        if any(item.iter_markers(name="requires_rocm")) and not rocm_available:
            item.add_marker(skip_no_rocm)
        if any(item.iter_markers(name="requires_rocm_dev")) and not rocm_dev_available:
            item.add_marker(skip_no_rocm_dev)
        if any(item.iter_markers(name="requires_ck")) and not ck_available:
            item.add_marker(skip_no_ck)
        if any(item.iter_markers(name="requires_rocwmma")) and not rocwmma_available:
            item.add_marker(skip_no_rocwmma)
        if any(item.iter_markers(name="requires_rdna4")) and not _is_rdna4(gfx_arch):
            item.add_marker(skip_rdna4 if rocm_available else skip_no_rocm)
        if any(item.iter_markers(name="requires_cdna3")) and not _is_cdna3(gfx_arch):
            item.add_marker(skip_cdna3 if rocm_available else skip_no_rocm)
        if (
            any(item.iter_markers(name="requires_rocm"))
            and rocm_available
            and not supported_arch
        ):
            item.add_marker(skip_unsupported)
        if any(item.iter_markers(name="requires_cutile")):
            item.add_marker(skip_legacy_cutile)
        if "timing_serial" in item.keywords and not timing_selected:
            item.add_marker(skip_timing)


@pytest.fixture
def tmp_cache_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolated temporary cache directory for each test.

    Sets SOLEXECBENCH_CACHE_PATH so every builder writes build artifacts into a
    fresh temp directory, preventing pollution between tests.
    """
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("SOLEXECBENCH_CACHE_PATH", str(cache_dir))
    return cache_dir
