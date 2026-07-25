from __future__ import annotations

from sol_execbench.core.platform.runtime import detect_rocm_version


def test_collect_rocm_version_file_uses_discovered_root_argument(tmp_path) -> None:
    root = tmp_path / "custom-rocm"
    version_file = root / ".info" / "version"
    version_file.parent.mkdir(parents=True)
    version_file.write_text("7.2.0\n", encoding="utf-8")

    assert detect_rocm_version(root=root) == "7.2.0"


def test_collect_rocm_version_file_uses_dev_version_fallback(tmp_path) -> None:
    root = tmp_path / "custom-rocm"
    version_file = root / ".info" / "version-dev"
    version_file.parent.mkdir(parents=True)
    version_file.write_text("7.2.0-dev\n", encoding="utf-8")

    assert detect_rocm_version(root=root) == "7.2.0-dev"
