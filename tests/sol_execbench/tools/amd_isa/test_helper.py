# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess

import pytest

from sol_execbench.tools.amd_isa.helper import ensure_helper


@pytest.mark.cpp
@pytest.mark.skipif(shutil.which("cmake") is None, reason="CMake is required")
def test_vendored_helper_builds_and_rejects_invalid_protocol(tmp_path: Path) -> None:
    helper = ensure_helper(tmp_path / "cache")

    result = subprocess.run(
        [str(helper)],
        input="not-json\n",
        capture_output=True,
        text=True,
        check=True,
    )

    response = json.loads(result.stdout)
    assert response["ok"] is False
    assert response["error"]["code"] == "invalid_request"
