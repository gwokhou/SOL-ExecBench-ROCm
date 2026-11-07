# SPDX-License-Identifier: Apache-2.0
"""Freeze a public 24-case development denominator without GPU execution."""

import argparse
import json
import math
from collections.abc import Callable
from pathlib import Path

from sol_execbench.core.data.definition import Definition
from sol_execbench.core.data.json_utils import atomic_write_json_value
from sol_execbench.core.data.schema_versions import BenchmarkArtifactSchema
from sol_execbench.core.data.solution_instance import Solution
from sol_execbench.core.data.workload import Workload
from sol_execbench.core.integrity import sha256_file, stable_json_checksum

SOURCE_ROOT = Path("problems/AMD_AKA/torch2hip")
SEED = 20261011
DTYPE_BYTES = {"float16": 2, "bfloat16": 2, "float32": 4}
# Preserve original checks. Added shapes are public development boundaries,
# never additions to a scored or held-out corpus.
FAMILIES = (
    ("elementwise", "l1n26_gelu", {"M": 257, "N": 1023}),
    ("reduction", "l1n47_sum_reduction", {"M": 127, "N": 4095}),
    ("normalization", "l1n36_rmsnorm", {"B": 127, "F": 1023}),
    ("softmax", "l1n23_softmax", None),
    ("gemm", "l1n8_matmul_irregular", {"M": 513, "K": 1025, "N": 255}),
    ("composite", "3267_doubled_matmul", {"M": 127, "K": 129, "N": 255}),
)


def _storage_bytes(definition: dict, axes: dict[str, int]) -> int:
    """Count inputs and output only, explicitly excluding allocator/workspace."""
    size = 0
    for spec in (
        *definition["inputs"].values(),
        *definition["outputs"].values(),
    ):
        shape = spec["shape"]
        if shape is not None:
            extents = [
                axes[axis] if axis in axes else int(axis) for axis in shape
            ]
            size += math.prod(extents) * DTYPE_BYTES[spec["dtype"]]
    return size


def _candidate(definition: dict) -> dict:
    source = definition["reference"]
    if "import torch\n" not in source:
        source = "import torch\n" + source
    source += "\nrun = torch.compile(run, fullgraph=True, dynamic=False)\n"
    return Solution.model_validate(
        {
            "schema_version": BenchmarkArtifactSchema.SOLUTION,
            "name": f"upgrade_compile_{definition['name']}",
            "definition": definition["name"],
            "author": "stack-upgrade-development",
            "spec": {
                "languages": ["pytorch"],
                "target_hardware": ["gfx1200"],
                "entry_point": "kernel.py::run",
                "dependencies": ["torch"],
                "destination_passing_style": False,
            },
            "sources": [{"path": "kernel.py", "content": source}],
        }
    ).model_dump(mode="json")


def freeze_tensors(definition: dict, workload: dict, path: Path) -> None:
    """Generate CPU tensors once; a seed is not portable across Torch releases."""
    import torch
    from safetensors.torch import save_file

    generator = torch.Generator(device="cpu").manual_seed(SEED)
    tensors = {}
    for name, spec in definition["inputs"].items():
        if workload["inputs"][name]["type"] == "scalar":
            continue
        shape = [
            workload["axes"][axis] if axis in workload["axes"] else int(axis)
            for axis in spec["shape"]
        ]
        tensors[name] = torch.randn(
            shape, dtype=getattr(torch, spec["dtype"]), generator=generator
        )
    save_file(tensors, path)


def _freeze_family(
    repo: Path, root: Path, family: tuple, tensor_writer: Callable
) -> list[dict]:
    label, name, boundary = family
    original = repo / SOURCE_ROOT / name
    definition = json.loads((original / "definition.json").read_text())
    Definition.model_validate(definition)
    rows = [
        json.loads(line)
        for line in (original / "workload.jsonl").read_text().splitlines()
        if line.strip()
    ]
    if boundary is not None:
        rows = [
            *rows[:3],
            {**rows[0], "axes": boundary, "uuid": f"upgrade-{name}-nonaligned"},
        ]
    if len(rows) != 4:
        raise ValueError(f"expected four cases for {name}")
    candidate = _candidate(definition)
    directory = root / name
    directory.mkdir()
    # Definition input order is part of its callable-signature contract.
    atomic_write_json_value(
        directory / "definition.json", definition, sort_keys=False
    )
    atomic_write_json_value(directory / "solution.json", candidate)
    result = []
    for index, row in enumerate(rows):
        Workload.model_validate(row)
        path = directory / f"workload-{index}.json"
        tensor_path = directory / f"inputs-{index}.safetensors"
        tensor_writer(definition, row, tensor_path)
        frozen_row = {
            **row,
            "inputs": {
                name: spec
                if spec["type"] == "scalar"
                else {
                    "type": "safetensors",
                    "path": tensor_path.name,
                    "tensor_key": name,
                }
                for name, spec in row["inputs"].items()
            },
        }
        Workload.model_validate(frozen_row)
        atomic_write_json_value(path, frozen_row)
        result.append(
            {
                "id": row["uuid"],
                "family": label,
                "problem": name,
                "axes": row["axes"],
                "seed": SEED,
                "tensors": tensor_path.relative_to(root).as_posix(),
                "tensors_sha256": sha256_file(tensor_path),
                "workload": path.relative_to(root).as_posix(),
                "definition": f"{name}/definition.json",
                "candidate": f"{name}/solution.json",
                "candidate_sha256": sha256_file(directory / "solution.json"),
                "minimum_tensor_storage_bytes": _storage_bytes(
                    definition, row["axes"]
                ),
                "source_kind": "added_public_boundary"
                if row["uuid"].startswith("upgrade-")
                else "existing_public_workload",
            }
        )
    return result


def freeze(
    repo: Path, root: Path, *, tensor_writer: Callable = freeze_tensors
) -> dict:
    """Write once; tensor_writer is the small-data seam for control-plane tests."""
    root.mkdir(parents=True, exist_ok=False)
    items = [
        item
        for family in FAMILIES
        for item in _freeze_family(repo, root, family, tensor_writer)
    ]
    sources = {
        str(SOURCE_ROOT / name / file): sha256_file(
            repo / SOURCE_ROOT / name / file
        )
        for _, name, _ in FAMILIES
        for file in ("definition.json", "workload.jsonl")
    }
    files = {
        p.relative_to(root).as_posix(): sha256_file(p)
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }
    payload = {
        "authority": "development_inputs_only",
        "items": items,
        "files": files,
        "source_files": sources,
        "denominator_sha256": stable_json_checksum(files),
        "candidate_policy": "same torch.compile fullgraph static-shape source on both stacks",
        "input_policy": "identical frozen CPU safetensors bytes on both stacks",
        "limitations": [
            "Not GPU qualification",
            "Not a frozen measurement protocol",
            "Storage lower bounds exclude intermediates, compiler and allocator",
            "No native-library, transpose or attention coverage claim",
        ],
    }
    atomic_write_json_value(root / "manifest.json", payload)
    return payload


def main() -> None:
    """Create one immutable public development input snapshot."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = freeze(args.repo, args.output)
    print(
        json.dumps(
            {
                "items": len(payload["items"]),
                "denominator_sha256": payload["denominator_sha256"],
            }
        )
    )


if __name__ == "__main__":
    main()
