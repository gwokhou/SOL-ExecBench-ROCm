from __future__ import annotations

from pathlib import Path

import pytest

from sol_execbench.core.process import exclusive_file_lock


def test_exclusive_file_lock_rejects_concurrent_holder(tmp_path: Path) -> None:
    lock_path = tmp_path / "audit.lock"

    with (
        exclusive_file_lock(lock_path),
        pytest.raises(RuntimeError, match="already held"),
        exclusive_file_lock(lock_path),
    ):
        pass

    with exclusive_file_lock(lock_path):
        assert lock_path.is_file()
