from __future__ import annotations

from sol_execbench.core.platform.dependency_matrix.collectors import (
    _collect_rocm_version_file,
)


def test_collect_rocm_version_file_uses_discovered_root_argument(tmp_path) -> None:
    root = tmp_path / "custom-rocm"
    version_file = root / ".info" / "version"
    version_file.parent.mkdir(parents=True)
    version_file.write_text("7.2.0\n", encoding="utf-8")

    assert _collect_rocm_version_file(root) == "7.2.0"


def test_collect_rocm_version_file_uses_dev_version_fallback(tmp_path) -> None:
    root = tmp_path / "custom-rocm"
    version_file = root / ".info" / "version-dev"
    version_file.parent.mkdir(parents=True)
    version_file.write_text("7.2.0-dev\n", encoding="utf-8")

    assert _collect_rocm_version_file(root) == "7.2.0-dev"
