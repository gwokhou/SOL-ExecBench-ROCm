"""Fixture builders shared by dataset inventory behavior tests."""

from __future__ import annotations

import json
from pathlib import Path


def default_definition(
    *,
    name: str = "matmul_forward",
    dtype: str = "float32",
    reference: str = "def run(x):\n    return x\n",
    custom_entrypoint: str | None = None,
) -> dict:
    payload = {
        "name": name,
        "description": "forward demo",
        "axes": {"N": {"type": "var"}},
        "inputs": {"x": {"shape": ["N"], "dtype": dtype}},
        "outputs": {"out": {"shape": ["N"], "dtype": dtype}},
        "reference": reference,
    }
    if custom_entrypoint:
        payload["custom_inputs_entrypoint"] = custom_entrypoint
    return payload


def workload(kind: str = "random", **extra: object) -> dict:
    spec: dict[str, object] = {"type": kind}
    spec.update(extra)
    return {"uuid": f"{kind}-w", "axes": {"N": 4}, "inputs": {"x": spec}}


def write_problem(
    root: Path,
    category: str,
    name: str,
    *,
    definition: dict | None = None,
    workloads: list[dict] | None = None,
    reference_file: bool = True,
    solution_file: bool = False,
) -> Path:
    problem_dir = root / category / name
    problem_dir.mkdir(parents=True)
    (problem_dir / "definition.json").write_text(
        json.dumps(definition or default_definition(name=name)) + "\n",
        encoding="utf-8",
    )
    (problem_dir / "workload.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in (workloads or [workload()])),
        encoding="utf-8",
    )
    if reference_file:
        (problem_dir / "reference.py").write_text(
            "def run(x):\n    return x\n", encoding="utf-8"
        )
    if solution_file:
        (problem_dir / "solution.json").write_text("{}\n", encoding="utf-8")
    return problem_dir


def write_solution(problem_dir: Path, name: str, payload: dict | str) -> None:
    text = (
        payload
        if isinstance(payload, str)
        else json.dumps(payload, sort_keys=True) + "\n"
    )
    (problem_dir / name).write_text(text, encoding="utf-8")
