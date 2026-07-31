from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from sol_execbench.core.platform.source_state import verify_git_source_state


def test_verifies_exact_clean_release_source(tmp_path: Path) -> None:
    repository, revision = _repository(tmp_path)

    state = verify_git_source_state(
        repository,
        expected_revision=revision,
        paths=("src",),
    )

    assert state.clean
    assert state.revision == revision


def test_rejects_wrong_source_revision(tmp_path: Path) -> None:
    repository, _ = _repository(tmp_path)

    with pytest.raises(ValueError, match="source revision mismatch"):
        verify_git_source_state(
            repository,
            expected_revision="a" * 40,
            paths=("src",),
        )


@pytest.mark.parametrize("untracked", [False, True])
def test_rejects_uncommitted_release_source(
    tmp_path: Path,
    *,
    untracked: bool,
) -> None:
    repository, revision = _repository(tmp_path)
    path = repository / "src" / ("new.py" if untracked else "tracked.py")
    path.write_text("changed = True\n", encoding="utf-8")

    with pytest.raises(ValueError, match="uncommitted changes"):
        verify_git_source_state(
            repository,
            expected_revision=revision,
            paths=("src",),
        )


def _repository(tmp_path: Path) -> tuple[Path, str]:
    repository = tmp_path / "repository"
    source = repository / "src"
    source.mkdir(parents=True)
    (source / "tracked.py").write_text("committed = True\n", encoding="utf-8")
    _git(repository, "init")
    _git(repository, "config", "user.email", "tests@example.invalid")
    _git(repository, "config", "user.name", "Release Test")
    _git(repository, "add", "src/tracked.py")
    _git(repository, "commit", "-m", "Initial source")
    return repository, _git(repository, "rev-parse", "HEAD").stdout.strip()


def _git(repository: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
