#!/usr/bin/env python3
"""Download the nvidia/SOL-ExecBench dataset from Hugging Face and unpack it
into the local ``data/SOL-ExecBench/benchmark/<subset>/<problem>/`` layout expected
by sol-execbench.

Each problem directory contains:
  - definition.json
  - reference.py
  - workload.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sol_execbench.core.dataset import (  # noqa: E402
    DatasetManifestSource,
    build_dataset_manifest,
    validate_categories,
    write_dataset_manifest,
)

REPO_ID = "nvidia/SOL-ExecBench"
OUTPUT_DIR = ROOT / "data" / "SOL-ExecBench" / "benchmark"


def _build_definition(row: dict) -> dict:
    """Assemble a definition.json dict from a dataset row."""
    definition: dict = {
        "name": row["name"],
        "hf_id": row.get("hf_id", None),
        "description": row["description"],
        "axes": json.loads(row["axes"]),
        "custom_inputs_entrypoint": row.get("custom_inputs_entrypoint", None),
        "inputs": json.loads(row["inputs"]),
        "outputs": json.loads(row["outputs"]),
        "reference": row["reference"],
    }
    return definition


def _write_text_if_changed(path: Path, content: str, *, force: bool = False) -> str:
    data = content.encode("utf-8")
    if path.exists():
        old_data = path.read_bytes()
        if old_data == data:
            return "unchanged"
        if not force:
            raise FileExistsError(
                f"{path} exists with different content; rerun with --force to overwrite"
            )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return "written"


def _write_manifest(
    output_root: Path,
    manifest_path: Path,
    *,
    categories: tuple[str, ...],
    revision: str | None,
    local_provenance: str | None = None,
) -> None:
    source = DatasetManifestSource(
        repo_id=REPO_ID,
        revision=revision,
        local_provenance=local_provenance,
    )
    manifest = build_dataset_manifest(
        output_root,
        categories=categories,
        source=source,
    )
    write_dataset_manifest(manifest, manifest_path)
    print(f"Manifest written to {manifest_path}")


def _process_subset(
    subset: str,
    *,
    output_root: Path,
    revision: str | None,
    force: bool = False,
) -> None:
    print(f"Downloading {REPO_ID} config={subset} ...")
    load_kwargs = {"name": subset, "split": "train"}
    if revision:
        load_kwargs["revision"] = revision
    ds = load_dataset(REPO_ID, **load_kwargs)

    written = 0
    unchanged = 0
    for row in ds:
        name = row["name"]
        problem_dir = output_root / subset / name

        # definition.json
        definition = _build_definition(row)
        status = _write_text_if_changed(
            problem_dir / "definition.json",
            json.dumps(definition, indent=4) + "\n",
            force=force,
        )
        written += status == "written"
        unchanged += status == "unchanged"

        # reference.py
        status = _write_text_if_changed(
            problem_dir / "reference.py",
            row["reference"],
            force=force,
        )
        written += status == "written"
        unchanged += status == "unchanged"

        # workload.jsonl
        workloads = json.loads(row["workloads"])
        workload_text = "".join(json.dumps(workload) + "\n" for workload in workloads)
        status = _write_text_if_changed(
            problem_dir / "workload.jsonl",
            workload_text,
            force=force,
        )
        written += status == "written"
        unchanged += status == "unchanged"

    print(
        f"  -> {len(ds)} problems inspected at {output_root / subset} "
        f"({written} files written, {unchanged} unchanged)"
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download or verify the public SOL-ExecBench dataset layout.",
    )
    parser.add_argument(
        "--category",
        action="append",
        dest="categories",
        help="Dataset category to process. Repeat for multiple categories. Defaults to all public categories.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUT_DIR,
        help="Dataset benchmark output root. Defaults to data/SOL-ExecBench/benchmark.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Optional path for a deterministic acquisition/layout manifest JSON.",
    )
    parser.add_argument(
        "--revision",
        help="Optional Hugging Face dataset revision to pass to datasets.load_dataset.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite divergent existing canonical dataset files.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Inspect local layout and optionally write a manifest without downloading.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    categories = validate_categories(args.categories)
    output_root = Path(args.output_root)

    if args.verify_only:
        if args.manifest:
            _write_manifest(
                output_root,
                Path(args.manifest),
                categories=categories,
                revision=args.revision,
                local_provenance="verify_only_local_layout",
            )
        manifest = build_dataset_manifest(
            output_root,
            categories=categories,
            source=DatasetManifestSource(
                repo_id=REPO_ID,
                revision=args.revision,
                local_provenance="verify_only_local_layout",
            ),
        )
        return 0 if manifest.claim_boundary.acquisition_or_layout_complete else 1

    try:
        for subset in categories:
            _process_subset(
                subset,
                output_root=output_root,
                revision=args.revision,
                force=args.force,
            )
    except Exception as exc:
        print(f"Download failed: {exc}", file=sys.stderr)
        if args.manifest:
            _write_manifest(
                output_root,
                Path(args.manifest),
                categories=categories,
                revision=args.revision,
                local_provenance="download_failed_local_layout",
            )
        return 1

    if args.manifest:
        _write_manifest(
            output_root,
            Path(args.manifest),
            categories=categories,
            revision=args.revision,
        )

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
