# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""GPU-gated equivalence validation for the AKA-derived problems.

Runs ``scripts/aka_equivalence_check.py`` against the committed corpus and
asserts every executable workload and every output is a faithful oracle.
Skipped on hosts without a ROCm GPU or without a local AKA clone.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPT = REPO_ROOT / "scripts" / "aka_equivalence_check.py"
AKA_HEAD = REPO_ROOT / "data" / "AgentKernelArena" / ".aka-head"


@pytest.mark.requires_rocm_gpu
@pytest.mark.requires_rdna4
@pytest.mark.skipif(not AKA_HEAD.is_file(), reason="requires a local AKA clone")
def test_aka_equivalence_check_passes_for_the_corpus():
    result = subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=REPO_ROOT,
        capture_output=True,
        check=False,
        text=True,
        timeout=600,
    )

    assert result.returncode == 0, (
        f"equivalence check failed:\n{result.stdout}\n{result.stderr}"
    )
    assert "[FAIL]" not in result.stdout, result.stdout
    assert "cross-check=failed" not in result.stdout, result.stdout
    assert "43 source-equivalent, 2 explicitly not applicable" in result.stdout
