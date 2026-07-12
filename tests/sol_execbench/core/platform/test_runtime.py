from __future__ import annotations

from sol_execbench.core.platform.runtime import hardware_from_device


def test_hardware_from_device_supports_cpu() -> None:
    assert hardware_from_device("cpu")
