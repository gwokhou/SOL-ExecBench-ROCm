# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""GPU-gated equivalence validation for the AKA-derived problems.

Runs ``scripts/aka_equivalence_check.py`` against the committed corpus and
asserts every authored reference is a sane, faithful oracle. Skipped on hosts
without a ROCm GPU or without a local AKA clone.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPT = REPO_ROOT / "scripts" / "aka_equivalence_check.py"
AKA_HEAD = REPO_ROOT / "data" / "AgentKernelArena" / ".aka-head"

_HAS_GPU = False
try:  # noqa: SIM105 — guard import side effects
    import torch

    _HAS_GPU = torch.cuda.is_available()
except Exception:  # noqa: BLE001
    _HAS_GPU = False


@pytest.mark.skipif(not _HAS_GPU, reason="requires a ROCm GPU")
@pytest.mark.skipif(not AKA_HEAD.is_file(), reason="requires a local AKA clone")
def test_aka_equivalence_check_passes_for_the_corpus():
    result = subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=600,
    )

    assert result.returncode == 0, (
        f"equivalence check failed:\n{result.stdout}\n{result.stderr}"
    )
    # Robust to corpus-size changes: every problem must pass sanity with no FAIL
    # line and zero numerical cross-check failures.
    assert "[FAIL]" not in result.stdout, result.stdout
    assert "0 failed" in result.stdout, result.stdout
