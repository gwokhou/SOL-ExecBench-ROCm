from __future__ import annotations

from pathlib import Path
import tomllib
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[4]
PYPROJECT = REPO_ROOT / "pyproject.toml"
PRE_COMMIT_CONFIG = REPO_ROOT / ".pre-commit-config.yaml"
PYTHON_VERSION = REPO_ROOT / ".python-version"
QUALITY_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "code-quality.yml"


def _quality_workflow() -> dict[str, Any]:
    return yaml.safe_load(QUALITY_WORKFLOW.read_text())


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


def test_python_support_is_pinned_to_3_12() -> None:
    data = tomllib.loads(PYPROJECT.read_text())
    workflow = _quality_workflow()

    assert data["project"]["requires-python"] == ">=3.12,<3.13"
    assert PYTHON_VERSION.read_text().strip() == "3.12"
    assert all(
        next(
            step
            for step in job["steps"]
            if str(step.get("uses", "")).startswith("actions/setup-python@")
        )["with"]["python-version-file"]
        == ".python-version"
        for job in workflow["jobs"].values()
    )
    assert "3.13" not in QUALITY_WORKFLOW.read_text()


def test_quality_workflow_splits_parallel_responsibilities() -> None:
    workflow = _quality_workflow()
    jobs = workflow["jobs"]

    assert workflow["permissions"] == {"contents": "read"}
    assert workflow["concurrency"]["cancel-in-progress"] is True
    assert set(jobs) == {"quality", "package-tests", "solar-tests"}
    assert all(job["timeout-minutes"] == 15 for job in jobs.values())

    for job in jobs.values():
        uses = [step["uses"] for step in job["steps"] if "uses" in step]
        assert uses == [
            "actions/checkout@v7.0.1",
            "actions/setup-python@v7.0.0",
            "astral-sh/setup-uv@v9.0.0",
        ]
        setup_uv = job["steps"][2]
        assert setup_uv["with"]["version"] == "0.11.31"
        assert setup_uv["with"]["enable-cache"] is True
        assert setup_uv["with"]["cache-dependency-glob"] == (
            "pyproject.toml\nuv.lock\n"
        )
        assert job["steps"][3]["run"] == "uv sync --locked --all-groups"

    quality_commands = "\n".join(
        step.get("run", "") for step in jobs["quality"]["steps"]
    )
    package_commands = "\n".join(
        step.get("run", "") for step in jobs["package-tests"]["steps"]
    )
    solar_commands = "\n".join(
        step.get("run", "") for step in jobs["solar-tests"]["steps"]
    )
    assert "ruff format --check ." in quality_commands
    assert "ty check" in quality_commands
    assert "tests/sol_execbench" in package_commands
    assert "tests/tools" in package_commands
    assert "tests/solar" not in package_commands
    assert "coverage run -m pytest -n 0 tests/solar" in solar_commands
    assert "tests/solar" not in quality_commands


def test_ci_and_docker_pin_the_same_uv_version() -> None:
    workflow = _quality_workflow()
    dockerfile = (REPO_ROOT / "docker" / "Dockerfile").read_text()

    assert all(
        job["steps"][2]["with"]["version"] == "0.11.31"
        for job in workflow["jobs"].values()
    )
    assert "ghcr.io/astral-sh/uv:0.11.31" in dockerfile
