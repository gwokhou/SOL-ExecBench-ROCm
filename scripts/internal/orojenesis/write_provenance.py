#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Write deterministic provenance for one pinned Orojenesis mapper build."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _git(home: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(home), *args],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return completed.stdout.strip()


def _archive_sha256(home: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(home), "archive", "--format=tar", "HEAD"],
        check=True,
        capture_output=True,
        timeout=60,
    )
    return hashlib.sha256(completed.stdout).hexdigest()


def _compiler_identity(wrapper: Path) -> str:
    completed = subprocess.run(
        [str(wrapper), "--version"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return completed.stdout.splitlines()[0].strip()


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--home", required=True, type=Path)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--expected-tree", required=True)
    parser.add_argument("--expected-archive-sha256", required=True)
    parser.add_argument("--builder-image", required=True)
    parser.add_argument("--ubuntu-snapshot", required=True)
    parser.add_argument("--source-date-epoch", required=True, type=int)
    parser.add_argument("--compiler-wrapper", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    home = args.home.resolve()
    mapper = home / "bin" / "timeloop-mapper"
    observed = {
        "commit": _git(home, "rev-parse", "HEAD"),
        "tree": _git(home, "rev-parse", "HEAD^{tree}"),
        "archive": _archive_sha256(home),
    }
    expected = {
        "commit": args.expected_commit,
        "tree": args.expected_tree,
        "archive": args.expected_archive_sha256,
    }
    if observed != expected:
        raise ValueError(f"Orojenesis source identity mismatch: {observed}")
    payload = {
        "schema_version": 1,
        "source": {
            "repository": "https://github.com/NVlabs/timeloop.git",
            "commit": observed["commit"],
            "tree_git_oid": observed["tree"],
            "archive_sha256": observed["archive"],
        },
        "artifact": {
            "path": "bin/timeloop-mapper",
            "sha256": _sha256(mapper),
        },
        "build": {
            "builder_image": args.builder_image,
            "ubuntu_snapshot": args.ubuntu_snapshot,
            "source_date_epoch": args.source_date_epoch,
            "compiler": _compiler_identity(args.compiler_wrapper),
            "compiler_wrapper_sha256": _sha256(args.compiler_wrapper),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=args.output.parent, delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
