"""Assemble a fresh, content-bound E2 execution bundle without opening inputs."""

import argparse
import re
from pathlib import Path
from shutil import copyfile
from typing import Any

from scripts.internal.solar.e2_successor_freeze import (
    _BUDGETS,
    _ROLE_NAMES,
    FREEZE_KIND,
    _verify_budget,
    _verify_capsule,
    _verify_interruptions,
    _verify_measured_capture,
    _verify_qualification,
    schedule_sha256,
)
from sol_execbench.core.data.json_utils import (
    atomic_write_json_value,
    load_json_value,
)
from sol_execbench.core.integrity import sha256_file
from sol_execbench.core.scoring.release_environment import (
    verify_release_source_state,
)

_FROZEN_DOCUMENTS = {
    "original_spec": "solar-m0-judge-preregistration-2026-11-07.md",
    "count_erratum": "solar-e0-cost-count-erratum-2026-11-08.md",
}
_FREEZE_PATH = "freeze.json"
_READINESS = "readiness"
_CAPSULE = "capsule"
_SOURCE_PACKAGE = Path("scripts/internal/solar")


def _check_frozen_inputs(base: dict[str, Any], paths: dict[str, Path]) -> None:
    if paths.keys() != _ROLE_NAMES:
        raise ValueError(
            "E2 launch bundle must bind all required readiness roles"
        )
    for role, name in _FROZEN_DOCUMENTS.items():
        original = base["bound_files"][f"/sol-execbench/docs/internal/{name}"]
        if sha256_file(paths[role]) != original:
            raise ValueError(f"E2 approved frozen input has changed: {role}")


def _snapshot(paths: dict[str, Path], root: Path) -> dict[str, Any]:
    readiness = root / _READINESS
    readiness.mkdir(exist_ok=False)
    references = {}
    for role, source in paths.items():
        path = readiness / Path(role).with_suffix(source.suffix)
        digest = sha256_file(source)
        copyfile(source, path)
        if sha256_file(path) != digest:
            raise ValueError(
                f"E2 readiness source changed during copying: {role}"
            )
        references[role] = {
            "path": path.relative_to(root).as_posix(),
            "sha256": digest,
        }
    return references


def _snapshot_sources(root: Path) -> dict[str, str]:
    target = root / _CAPSULE / _SOURCE_PACKAGE
    target.mkdir(parents=True, exist_ok=False)
    digests = {}
    for script in sorted(Path(__file__).resolve().parent.glob("e2_*.py")):
        copy = target / script.name
        digest = sha256_file(script)
        copyfile(script, copy)
        if sha256_file(copy) != digest:
            raise ValueError(
                f"E2 research source changed during copying: {script.name}"
            )
        digests[(_SOURCE_PACKAGE / script.name).as_posix()] = digest
    return digests


def prepare_bundle(
    bundle: Path,
    *,
    repository: Path,
    base_freeze: Path,
    paths: dict[str, Path],
    activity_directory: str,
) -> Path:
    """Record exact code and measured prerequisite bytes in a new bundle."""
    if re.fullmatch(r"[a-z0-9][a-z0-9-]+", activity_directory) is None:
        raise ValueError("E2 successor requires one frozen activity directory")
    base = load_json_value(base_freeze)
    _check_frozen_inputs(base, paths)
    execution = base["execution_commit"]
    verify_release_source_state(repository, expected_revision=execution)
    _verify_qualification(paths["b2_qualification"], execution)
    _verify_budget(paths)
    _verify_interruptions(paths)
    bundle.mkdir(exist_ok=False)
    references = _snapshot(paths, bundle)
    source_digests = _snapshot_sources(bundle)
    spec = {
        "research_format": FREEZE_KIND,
        "activity_directory": activity_directory,
        "execution_commit": execution,
        "image_id": base["image_id"],
        "device": base["device"],
        "container_semantic_runtime": base["container_semantic_runtime"],
        "worker_memory_bytes": base["budgets"]["worker_memory_bytes"],
        "cpu_affinity": base["budgets"]["required_cpu_affinity"],
        "schedule_sha256": schedule_sha256(),
        "budgets": _BUDGETS,
        "capsule_sha256": source_digests,
        "evidence_bindings": references,
        "performance_authority": False,
    }
    copied = {role: bundle / ref["path"] for role, ref in references.items()}
    _check_frozen_inputs(base, copied)
    _verify_budget(copied)
    _verify_interruptions(copied)
    _verify_qualification(copied["b2_qualification"], execution)
    _verify_capsule(bundle / _CAPSULE, source_digests)
    _verify_measured_capture(copied, spec)
    frozen = bundle / _FREEZE_PATH
    atomic_write_json_value(frozen, spec)
    return frozen


def main() -> None:
    """Prepare and print the exact freeze digest without running any workloads."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle", "repository", "base-freeze", "bindings"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--activity-directory", required=True)
    args = parser.parse_args()
    bindings = load_json_value(args.bindings)
    if any(not Path(path).is_absolute() for path in bindings.values()):
        parser.error("E2 readiness bindings require explicit absolute paths")
    frozen = prepare_bundle(
        args.bundle,
        repository=args.repository,
        base_freeze=args.base_freeze,
        paths={role: Path(path) for role, path in bindings.items()},
        activity_directory=args.activity_directory,
    )
    print(sha256_file(frozen))


if __name__ == "__main__":
    main()
