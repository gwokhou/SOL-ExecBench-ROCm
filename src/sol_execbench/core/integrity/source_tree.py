"""Bound executable source and packaged contracts without consulting Git state."""

from pathlib import Path

from sol_execbench.core.integrity.checksums import sha256_file

_SOURCE_SUFFIXES = frozenset(
    {".py", ".json", ".yaml", ".yml", ".cpp", ".h", ".hpp", ".hip"}
)
_MAX_FILES = 8192
_MAX_BYTES = 64 * 1024**2


def source_tree_manifest(root: Path) -> dict[str, str]:
    """Hash a bounded source tree, rejecting symlinks and excessive content."""
    paths: list[Path] = []
    size = 0
    for path in root.rglob("*"):
        if path.suffix not in _SOURCE_SUFFIXES or not path.is_file():
            continue
        if path.is_symlink():
            raise ValueError("source identity does not accept symlinks")
        size += path.stat().st_size
        paths.append(path)
        if len(paths) > _MAX_FILES or size > _MAX_BYTES:
            raise ValueError("source identity budget exceeded")
    if not paths:
        raise ValueError("source tree is empty")
    return {
        str(path.relative_to(root)): sha256_file(path) for path in sorted(paths)
    }
