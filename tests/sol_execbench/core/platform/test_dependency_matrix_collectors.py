from __future__ import annotations

import pytest

from sol_execbench.core.platform.runtime import detect_rocm_version


def test_collect_rocm_version_file_uses_discovered_root_argument(
    tmp_path,
) -> None:
    root = tmp_path / "custom-rocm"
    version_file = root / ".info" / "version"
    version_file.parent.mkdir(parents=True)
    version_file.write_text("10.1.0\n", encoding="utf-8")

    assert detect_rocm_version(root=root) == "10.1.0"


def test_collect_rocm_version_file_uses_dev_version_fallback(tmp_path) -> None:
    root = tmp_path / "custom-rocm"
    version_file = root / ".info" / "version-dev"
    version_file.parent.mkdir(parents=True)
    version_file.write_text("10.1.0-dev\n", encoding="utf-8")

    assert detect_rocm_version(root=root) == "10.1.0-dev"


@pytest.mark.parametrize("version_name", ["version", "version-dev"])
def test_collect_rocm_version_from_modular_core(tmp_path, version_name) -> None:
    root = tmp_path / "wheel-sdk"
    version_file = root / "core" / ".info" / version_name
    version_file.parent.mkdir(parents=True)
    version_file.write_text("10.1.0\n", encoding="utf-8")

    assert detect_rocm_version(root=root) == "10.1.0"
