#!/usr/bin/env python3
"""Inspect a SOL-ExecBench dataset root and write inventory/readiness sidecars."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sol_execbench.core.dataset import (  # noqa: E402
    build_dataset_inventory,
    build_ready_subset,
    classify_rocm_readiness,
    validate_categories,
    write_dataset_inventory,
    write_dataset_readiness,
    write_ready_subset,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate SOL-ExecBench dataset inventory/readiness sidecars.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--readiness", type=Path)
    parser.add_argument("--ready-subset", type=Path)
    parser.add_argument("--category", action="append", dest="categories")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    categories = validate_categories(args.categories)
    inventory = build_dataset_inventory(
        args.dataset_root,
        categories=categories,
        manifest_path=args.manifest,
    )
    readiness = classify_rocm_readiness(inventory, dataset_root=args.dataset_root)
    subset = build_ready_subset(readiness, dataset_root=args.dataset_root)
    if args.inventory:
        write_dataset_inventory(inventory, args.inventory)
    if args.readiness:
        write_dataset_readiness(readiness, args.readiness)
    if args.ready_subset:
        write_ready_subset(subset, args.ready_subset)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
