# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from io import BytesIO
import hashlib
import zipfile

import pytest

from sol_execbench.tools.amd_isa.errors import IsaSpecUnavailableError
from sol_execbench.tools.amd_isa.repository import IsaSpecRepository


def _entry(payload: bytes) -> dict[str, object]:
    archive = BytesIO()
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("amdgpu_isa_rdna4.xml", payload)
    archive_bytes = archive.getvalue()
    return {
        "url": "https://example.invalid/isa.zip",
        "archive_sha256": hashlib.sha256(archive_bytes).hexdigest(),
        "files": {
            "rdna4": {
                "name": "amdgpu_isa_rdna4.xml",
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        },
    }


def test_architecture_family_normalizes_gfx_features() -> None:
    assert IsaSpecRepository.architecture_family("gfx1200:xnack-") == "rdna4"
    assert IsaSpecRepository.architecture_family("gfx942") == "cdna3"
    assert IsaSpecRepository.architecture_family("rdna4") == "rdna4"


def test_unknown_architecture_is_explicitly_unavailable() -> None:
    with pytest.raises(IsaSpecUnavailableError, match="no machine-readable"):
        IsaSpecRepository.architecture_family("gfx9999")


def test_repository_downloads_once_and_checks_extracted_files(
    tmp_path, monkeypatch
) -> None:
    payload = b"<isa>fixture</isa>"
    entry = _entry(payload)
    archive = BytesIO()
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("amdgpu_isa_rdna4.xml", payload)
    calls = 0

    def open_fixture(*_: object, **__: object) -> BytesIO:
        nonlocal calls
        calls += 1
        return BytesIO(archive.getvalue())

    monkeypatch.setattr("sol_execbench.tools.amd_isa.repository.urlopen", open_fixture)
    repository = IsaSpecRepository(cache_root=tmp_path / "cache")
    repository._releases = {"fixture": entry}
    repository._default_release = "fixture"

    assert repository.spec_path("gfx1200").read_bytes() == payload
    assert repository.spec_path("gfx1200").read_bytes() == payload
    assert calls == 1


def test_offline_repository_never_downloads(tmp_path) -> None:
    repository = IsaSpecRepository(cache_root=tmp_path / "cache")

    with pytest.raises(IsaSpecUnavailableError, match="downloads are disabled"):
        repository.spec_path("gfx1200", allow_download=False)
