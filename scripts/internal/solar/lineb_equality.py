"""B1 equality campaign: fake backend graphs vs pristine real backend.

Runs make_fx extraction per scored workload on two trees (pristine HEAD for
the real side, this worktree for the fake side), then compares the
operator_graph.yaml sha256 per item. CPU, per-slot subprocesses.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

WORKTREE = Path(__file__).resolve().parents[3]
PRISTINE = Path("/home/guohao/PyCharmMiscProject/SOL-ExecBench-ROCm")


def _extract_one(tree: Path, item: str, out: Path, log: Path) -> str | None:
    """Extract one slot on the given tree and return its graph digest."""
    from sol_execbench.core.process.subprocesses import (
        run_in_process_group_to_files,
    )

    out.mkdir(parents=True, exist_ok=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    record = out / "graph.sha"
    if record.is_file():
        return record.read_text().strip() or None
    command = [
        sys.executable,
        str(WORKTREE / "scripts/internal/solar/lineb_extract_slot.py"),
        "--repo",
        str(tree),
        "--item",
        item,
        "--output",
        str(out),
    ]
    import os

    env = None
    if tree is not PRISTINE:
        env = {
            **os.environ,
            "PYTHONPATH": str(tree / "src"),
        }
    else:
        env = dict(os.environ)
    try:
        run_in_process_group_to_files(
            command,
            log.with_suffix(".out"),
            log.with_suffix(".err"),
            cwd=tree,
            timeout=900.0,
            env=env,
        )
    except subprocess.TimeoutExpired:
        record.write_text("TIMEOUT")
        return "TIMEOUT"
    return record.read_text().strip() or None


def main() -> int:
    """Run both sides over the corpus and write the equality matrix."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    items: list[str] = []
    manifest = Path("problems/AMD_AKA/manifest.yaml")
    from sol_execbench.core.dataset.aka_corpus import (
        AKACorpusManifest,
        AKACorpusRole,
    )

    corpus = AKACorpusManifest.load(manifest)
    for entry in corpus.entries:
        if entry.role is not AKACorpusRole.SCORED:
            continue
        problem = entry.relative_problem_dir.as_posix()
        for workload_uuid in entry.workload_uuids:
            items.append(f"{problem}/{workload_uuid}")
    rows = []
    equal = 0
    for index, item in enumerate(items):
        real = _extract_one(
            PRISTINE,
            item,
            args.output / "real" / item,
            args.output / "logs" / f"r{index}",
        )
        fake = _extract_one(
            WORKTREE,
            item,
            args.output / "fake" / item,
            args.output / "logs" / f"f{index}",
        )
        same = real == fake and real not in (None, "TIMEOUT")
        equal += same
        rows.append({"item": item, "real": real, "fake": fake, "equal": same})
        if index % 20 == 0:
            print(f"{index}/{len(items)} equal={equal}", flush=True)
    with (args.output / "equality-matrix.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    summary = {
        "schema_version": "sol_execbench.solar_lineb_equality.v1",
        "total": len(rows),
        "equal": equal,
        "mismatches": [r for r in rows if not r["equal"]][:32],
    }
    (args.output / "lineb-equality-summary.json").write_text(
        json.dumps(summary, indent=1, sort_keys=True) + "\n"
    )
    print(json.dumps({"total": len(rows), "equal": equal}))
    return 0 if equal == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
