from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src" / "sol_execbench"

HIP_AGENT_SEMANTICS = [
    "hip_agent",
    "provider turn",
    "provider turns",
    "agent prompt",
    "agent prompts",
    "search state",
    "lineage",
    "checkpoint",
    "checkpoints",
    "run manifest",
    "run manifests",
    "validation verdict",
    "validation verdicts",
]


def _source_files() -> list[Path]:
    return [
        path
        for path in SRC_ROOT.rglob("*.py")
        if "__pycache__" not in path.parts
    ]


def _imports_hip_agent(path: Path) -> bool:
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name == "hip_agent" for alias in node.names):
                return True
        if isinstance(node, ast.ImportFrom) and node.module == "hip_agent":
            return True
    return False


def test_sol_does_not_import_hip_agent_package() -> None:
    offenders = [
        str(path.relative_to(REPO_ROOT))
        for path in _source_files()
        if _imports_hip_agent(path)
    ]

    assert not offenders, (
        "SOL must not own HIP agent/run semantics or import hip_agent. "
        f"Offenders: {offenders}"
    )


def test_sol_source_does_not_absorb_hip_agent_run_semantics() -> None:
    offenders: list[str] = []
    for path in _source_files():
        lowered = path.read_text().lower()
        matches = [term for term in HIP_AGENT_SEMANTICS if term in lowered]
        if matches:
            offenders.append(f"{path.relative_to(REPO_ROOT)}: {matches}")

    assert not offenders, (
        "SOL must not own HIP agent/run semantics such as prompts, provider "
        f"turns, search state, lineage, checkpoints, run manifests, or HIP "
        f"validation verdict ownership. Offenders: {offenders}"
    )
