from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[4]
PYPROJECT = REPO_ROOT / "pyproject.toml"
PRE_COMMIT_CONFIG = REPO_ROOT / ".pre-commit-config.yaml"
PYTHON_VERSION = REPO_ROOT / ".python-version"
QUALITY_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "code-quality.yml"
CHECKOUT_ACTION = "actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1"
SETUP_PYTHON_ACTION = (
    "actions/setup-python@5fda3b95a4ea91299a34e894583c3862153e4b97"
)
SETUP_UV_ACTION = "astral-sh/setup-uv@c771a70e6277c0a99b617c7a806ffedaca235ff9"


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
    python_jobs = [
        job
        for job in workflow["jobs"].values()
        if any(
            str(step.get("uses", "")).startswith("actions/setup-python@")
            for step in job["steps"]
        )
    ]
    assert python_jobs
    assert all(
        next(
            step
            for step in job["steps"]
            if str(step.get("uses", "")).startswith("actions/setup-python@")
        )["with"]["python-version-file"]
        == ".python-version"
        for job in python_jobs
    )
    assert "3.13" not in QUALITY_WORKFLOW.read_text()


def test_pytest_configuration_is_strict_and_scoped() -> None:
    data = tomllib.loads(PYPROJECT.read_text())
    pytest_config = data["tool"]["pytest"]["ini_options"]

    assert "pytest>=9.1.1" in data["dependency-groups"]["dev"]
    assert pytest_config["minversion"] == "9.1.1"
    assert pytest_config["testpaths"] == ["tests"]
    assert pytest_config["norecursedirs"] == ["data", "out"]
    assert pytest_config["required_plugins"] == ["pytest-xdist>=3.5"]
    assert pytest_config["strict_config"] is True
    assert pytest_config["strict_markers"] is True
    assert pytest_config["strict_parametrization_ids"] is True
    assert pytest_config["strict_xfail"] is True


def test_quality_workflow_splits_parallel_responsibilities() -> None:
    workflow = _quality_workflow()
    jobs = workflow["jobs"]

    assert workflow["permissions"] == {"contents": "read"}
    assert workflow["concurrency"]["cancel-in-progress"] is True
    assert set(jobs) == {
        "dependency-review",
        "quality",
        "package-tests",
        "default-parallel-tests",
        "macos-development",
        "solar-tests",
    }

    python_job_names = set(jobs) - {"dependency-review"}
    for name in python_job_names:
        job = jobs[name]
        uses = [step["uses"] for step in job["steps"] if "uses" in step]
        assert uses == [
            CHECKOUT_ACTION,
            SETUP_PYTHON_ACTION,
            SETUP_UV_ACTION,
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
    parallel_commands = "\n".join(
        step.get("run", "") for step in jobs["default-parallel-tests"]["steps"]
    )
    assert "pytest tests/" in parallel_commands
    macos_commands = "\n".join(
        step.get("run", "") for step in jobs["macos-development"]["steps"]
    )
    assert "ty check" in macos_commands
    assert "tests/sol_execbench/core/process" in macos_commands


def test_ci_and_docker_pin_the_same_uv_version() -> None:
    workflow = _quality_workflow()
    dockerfile = (REPO_ROOT / "docker" / "Dockerfile").read_text()

    setup_steps = [
        step
        for job in workflow["jobs"].values()
        for step in job["steps"]
        if str(step.get("uses", "")).startswith("astral-sh/setup-uv@")
    ]
    assert setup_steps
    assert all(step["with"]["version"] == "0.11.31" for step in setup_steps)
    assert "ghcr.io/astral-sh/uv:0.11.31" in dockerfile
