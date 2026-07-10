# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Lazy, isolated execution support for the system ROCm Compute Profiler."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator, Mapping, Sequence

from sol_execbench.core.scoring.hardware_calibration.models import CalibrationCandidate


ProfilerRunner = Callable[..., object]


@dataclass(frozen=True)
class ProfilerDiscovery:
    """Injected process and filesystem seam for profiler environment management."""

    tool_path: Path
    tool_version: str
    requirements_path: Path
    artifact_root: Path
    interpreter_abi: str
    run: ProfilerRunner = subprocess.run
    exists: Callable[[Path], bool] = Path.exists
    is_executable: Callable[[Path], bool] = lambda path: os.access(path, os.X_OK)
    read_text: Callable[[Path], str] = lambda path: path.read_text(encoding="utf-8")
    write_text: Callable[[Path, str], object] = lambda path, text: path.write_text(
        text, encoding="utf-8"
    )
    environ: Mapping[str, str] = field(default_factory=lambda: dict(os.environ))


@dataclass(frozen=True)
class ProfilerEnvironment:
    """The managed environment and explicit backend availability evidence."""

    state: str
    reason_code: str | None
    tool_path: Path
    tool_version: str
    requirements_sha256: str | None
    venv_path: Path
    interpreter_path: Path
    installed_distributions: tuple[str, ...] = ()


@dataclass(frozen=True)
class RooflineParseResult:
    """Unknown-only roofline parsing result with raw-output provenance."""

    candidates: tuple[CalibrationCandidate, ...]
    raw_output_sha256: str


def default_profiler_discovery(project_root: Path) -> ProfilerDiscovery:
    """Derive launcher, version, and requirements from the installed tool.

    ROCm's package layout changes between releases, so no release directory or
    profiler version is baked into the calibration contract.
    """
    tool_path = Path(shutil.which("rocprof-compute") or "/usr/bin/rocprof-compute")
    requirements_path = _find_requirements(tool_path)
    tool_version = _discover_tool_version(tool_path)
    return ProfilerDiscovery(
        tool_path=tool_path,
        tool_version=tool_version,
        requirements_path=requirements_path,
        artifact_root=project_root / ".artifacts" / "rocprof-compute",
        interpreter_abi=f"cp{sys.version_info.major}{sys.version_info.minor}",
    )


def _find_requirements(tool_path: Path) -> Path:
    """Find the requirements shipped beside the resolved ROCm installation."""
    try:
        resolved = tool_path.resolve(strict=False)
    except OSError:
        resolved = tool_path
    roots = (resolved.parent, *resolved.parents)
    candidates: list[Path] = []
    for root in roots:
        candidates.extend(
            (
                root / "requirements.txt",
                root / "libexec" / "rocprofiler-compute" / "requirements.txt",
                root / "lib" / "rocprofiler-compute" / "requirements.txt",
            )
        )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    # Preserve a deterministic unavailable path for diagnostics.
    return resolved.parent / "requirements.txt"


def _discover_tool_version(tool_path: Path) -> str:
    try:
        completed = subprocess.run(
            (str(tool_path), "--version"), capture_output=True, text=True, check=False
        )
    except OSError:
        return "unknown"
    output = f"{completed.stdout}\n{completed.stderr}"
    import re

    match = re.search(
        r"(?:version\s*)?(\d+(?:\.\d+){1,3}(?:[-+][\w.]+)?)", output, re.I
    )
    return match.group(1) if completed.returncode == 0 and match else "unknown"


def _requirements_digest(discovery: ProfilerDiscovery) -> str | None:
    try:
        return hashlib.sha256(
            discovery.read_text(discovery.requirements_path).encode()
        ).hexdigest()
    except OSError:
        return None


def _venv_path(discovery: ProfilerDiscovery, requirements_sha256: str) -> Path:
    root = discovery.artifact_root
    if root.name != "rocprof-compute" or root.parent.name != ".artifacts":
        raise ValueError(
            "profiler environments must live below .artifacts/rocprof-compute"
        )
    key = f"{discovery.tool_version}-{requirements_sha256}-{discovery.interpreter_abi}"
    return root / key / "venv"


def _manifest_path(venv_path: Path) -> Path:
    return venv_path / "installed-distributions.json"


def _environment_is_ready(discovery: ProfilerDiscovery, venv_path: Path) -> bool:
    # A manifest ties the environment to its explicit installation, preventing
    # an ambient package from being interpreted as managed profiler support.
    return discovery.exists(venv_path / "bin" / "python") and discovery.exists(
        _manifest_path(venv_path)
    )


def _tool_is_available(discovery: ProfilerDiscovery) -> bool:
    return discovery.exists(discovery.tool_path) and discovery.is_executable(
        discovery.tool_path
    )


def _load_manifest(discovery: ProfilerDiscovery, venv_path: Path) -> tuple[str, ...]:
    payload = json.loads(discovery.read_text(_manifest_path(venv_path)))
    if not isinstance(payload, list) or not all(
        isinstance(item, str) for item in payload
    ):
        raise ValueError("installed-distribution manifest must be a list of strings")
    return tuple(payload)


def parse_roofline_metrics(
    raw_output: str,
    metric_to_candidate: Mapping[str, tuple[str, str]],
) -> RooflineParseResult:
    """Represent untrusted CSV metrics as unknown rather than fabricated peaks.

    A profiler row alone cannot satisfy the calibration sample/stability contract,
    so even recognised labels are not converted into numeric calibration values.
    """
    digest = hashlib.sha256(raw_output.encode("utf-8")).hexdigest()
    try:
        rows = tuple(csv.DictReader(io.StringIO(raw_output)))
    except csv.Error:
        rows = ()
    seen_metrics = {
        str(row.get("Metric", "")).strip() for row in rows if row.get("Metric")
    }
    candidates: list[CalibrationCandidate] = []
    for metric, (key, _unit) in metric_to_candidate.items():
        reason = (
            "rocprof_compute_metric_requires_calibration_samples"
            if metric in seen_metrics
            else "rocprof_compute_metric_missing"
        )
        candidates.append(
            CalibrationCandidate(
                key=key,
                state="unknown",
                value=None,
                unit=None,
                reason_code=reason,
            )
        )
    for metric in sorted(seen_metrics - set(metric_to_candidate)):
        metric_digest = hashlib.sha256(metric.encode("utf-8")).hexdigest()[:12]
        candidates.append(
            CalibrationCandidate(
                key=f"rocprof_compute.unrecognised.{metric_digest}",
                state="unknown",
                value=None,
                unit=None,
                reason_code="rocprof_compute_metric_unrecognised",
            )
        )
    return RooflineParseResult(tuple(candidates), digest)


@contextmanager
def _file_lock(path: Path) -> Iterator[None]:
    """Use a process lock on POSIX hosts; calibration is Linux/ROCm-only."""
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _run(discovery: ProfilerDiscovery, command: Sequence[str]) -> object:
    return discovery.run(list(command), check=True)


def _installed_manifest(discovery: ProfilerDiscovery, python: Path) -> tuple[str, ...]:
    result = _run(discovery, (str(python), "-m", "pip", "freeze", "--all"))
    stdout = getattr(result, "stdout", "") or ""
    return tuple(sorted(line for line in str(stdout).splitlines() if line.strip()))


def _unknown(
    discovery: ProfilerDiscovery, reason_code: str, requirements_sha256: str | None
) -> ProfilerEnvironment:
    venv_path = _venv_path(discovery, requirements_sha256 or "unavailable")
    return ProfilerEnvironment(
        state="unknown",
        reason_code=reason_code,
        tool_path=discovery.tool_path,
        tool_version=discovery.tool_version,
        requirements_sha256=requirements_sha256,
        venv_path=venv_path,
        interpreter_path=venv_path / "bin" / "python",
    )


def _measured_or_unknown(
    discovery: ProfilerDiscovery,
    requirements_sha256: str,
    venv_path: Path,
) -> ProfilerEnvironment:
    if not _tool_is_available(discovery):
        return _unknown(
            discovery, "rocprof_compute_tool_unavailable", requirements_sha256
        )
    try:
        manifest = _load_manifest(discovery, venv_path)
    except OSError:
        return _unknown(
            discovery, "rocprof_compute_manifest_unavailable", requirements_sha256
        )
    except (TypeError, ValueError):
        return _unknown(
            discovery, "rocprof_compute_manifest_invalid", requirements_sha256
        )
    python = venv_path / "bin" / "python"
    return ProfilerEnvironment(
        state="measured",
        reason_code=None,
        tool_path=discovery.tool_path,
        tool_version=discovery.tool_version,
        requirements_sha256=requirements_sha256,
        venv_path=venv_path,
        interpreter_path=python,
        installed_distributions=manifest,
    )


def ensure_profiler_environment(
    discovery: ProfilerDiscovery, offline: bool, auto_install: bool
) -> ProfilerEnvironment:
    """Return managed profiler availability; never install without explicit consent."""
    requirements_sha256 = _requirements_digest(discovery)
    if requirements_sha256 is None:
        return _unknown(discovery, "rocprof_compute_requirements_unavailable", None)
    venv_path = _venv_path(discovery, requirements_sha256)
    python = venv_path / "bin" / "python"
    if not _tool_is_available(discovery):
        return _unknown(
            discovery, "rocprof_compute_tool_unavailable", requirements_sha256
        )
    if _environment_is_ready(discovery, venv_path):
        return _measured_or_unknown(discovery, requirements_sha256, venv_path)
    if offline:
        return _unknown(
            discovery,
            "rocprof_compute_dependencies_unavailable_offline",
            requirements_sha256,
        )
    if not auto_install:
        return _unknown(
            discovery,
            "rocprof_compute_dependencies_unavailable_no_auto_install",
            requirements_sha256,
        )
    try:
        with _file_lock(venv_path.parent / ".lock"):
            if not _environment_is_ready(discovery, venv_path):
                venv_path.parent.mkdir(parents=True, exist_ok=True)
                _run(discovery, ("uv", "venv", str(venv_path)))
                _run(
                    discovery,
                    (
                        "uv",
                        "pip",
                        "install",
                        "--python",
                        str(python),
                        "-r",
                        str(discovery.requirements_path),
                    ),
                )
                manifest = _installed_manifest(discovery, python)
                discovery.write_text(
                    _manifest_path(venv_path), json.dumps(manifest, indent=2) + "\n"
                )
    except (OSError, subprocess.SubprocessError, ValueError):
        return _unknown(
            discovery,
            "rocprof_compute_dependencies_install_failed",
            requirements_sha256,
        )
    return _measured_or_unknown(discovery, requirements_sha256, venv_path)


def run_rocprof_compute_bench_only(
    environment: ProfilerEnvironment,
    *,
    run: ProfilerRunner = subprocess.run,
    extra_args: Sequence[str] = (),
) -> object | None:
    """Run the system launcher with only the managed Python resolution changed."""
    if environment.state != "measured":
        return None
    env = dict(os.environ)
    env["PATH"] = f"{environment.venv_path / 'bin'}{os.pathsep}{env.get('PATH', '')}"
    env["VIRTUAL_ENV"] = str(environment.venv_path)
    env["PYTHONNOUSERSITE"] = "1"
    command = [str(environment.tool_path), "profile", "--bench-only", *extra_args]
    return run(command, env=env)
