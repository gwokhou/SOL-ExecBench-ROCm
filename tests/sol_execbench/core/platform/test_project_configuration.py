from __future__ import annotations

from pathlib import Path
import tomllib


REPO_ROOT = Path(__file__).resolve().parents[4]
PYPROJECT = REPO_ROOT / "pyproject.toml"
PRE_COMMIT_CONFIG = REPO_ROOT / ".pre-commit-config.yaml"


def test_ruff_preserves_excludes_for_explicit_hook_paths() -> None:
    data = tomllib.loads(PYPROJECT.read_text())

    assert data["tool"]["ruff"]["force-exclude"] is True


def test_pre_commit_hooks_use_locked_uv_runs() -> None:
    config = PRE_COMMIT_CONFIG.read_text()

    assert "entry: uv run --locked ruff check --fix" in config
    assert "entry: uv run --locked ruff format" in config
    assert "entry: uv run --locked ty check" in config
    assert "entry: uv run ruff" not in config
    assert "entry: uv run ty check" not in config
