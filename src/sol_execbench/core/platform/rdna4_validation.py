# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Content-addressed evidence for local gfx1200 hardware validation."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast
from xml.etree import ElementTree

from pydantic import ConfigDict, Field

from sol_execbench.core.data.base_model import (
    CurrentSchemaModel,
    StrictArtifactModel,
)
from sol_execbench.core.integrity import sha256_file, stable_json_checksum
from sol_execbench.core.integrity.schema_versions import (
    RDNA4_VALIDATION_SCHEMA_VERSION,
)

RDNA4_VALIDATION_GFX_TARGET = "gfx1200"
RDNA4_VALIDATION_PCI_VENDOR_ID = "0x1002"
RDNA4_VALIDATION_PCI_DEVICE_ID = "0x7590"
RDNA4_VALIDATION_ROCM_VERSION = "7.2.0"
RDNA4_VALIDATION_TORCH_VERSION = "2.11.0+rocm7.2"
RDNA4_VALIDATION_HIP_VERSION = "7.2.26015"
RDNA4_VALIDATION_TRITON_VERSION = "3.6.0"
_REQUIRED_ARTIFACTS = frozenset(
    {
        "environment-doctor.json",
        "pytest-rdna4.xml",
        "pytest.stderr.txt",
        "pytest.stdout.txt",
    },
)
_LOCAL_ATTESTATION_KINDS = frozenset(
    {"github_actions_self_hosted", "local_unsigned"},
)
_ROCM_PATH_VERSION = re.compile(r"(?:^|/)rocm-(\d+\.\d+\.\d+)(?:/|$)")

_VALIDATION_CONFIG = ConfigDict(extra="forbid", frozen=True)


class _ValidationModel(StrictArtifactModel):
    model_config = _VALIDATION_CONFIG


class Rdna4TargetSchema(_ValidationModel):
    """Validated hardware and user-space target identity."""

    gfx_target: str
    device_name: str
    device_index: int | None
    rocm_version: str
    torch_version: str
    hip_version: str
    pci_vendor_id: str
    pci_device_id: str


class Rdna4PytestSummary(_ValidationModel):
    """Aggregate pytest result bound by the validation manifest."""

    returncode: int
    tests: int = Field(ge=0)
    failures: int = Field(ge=0)
    errors: int = Field(ge=0)
    skipped: int = Field(ge=0)


class Rdna4Artifact(_ValidationModel):
    """Content-addressed validation artifact."""

    path: str
    size_bytes: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class Rdna4Attestation(_ValidationModel):
    """Explicitly local, non-release validation authority."""

    kind: Literal["github_actions_self_hosted", "local_unsigned"]
    trusted_execution: bool


class Rdna4ValidationManifest(CurrentSchemaModel):
    """Current local RDNA4 validation manifest."""

    model_config = _VALIDATION_CONFIG
    current_schema_version = RDNA4_VALIDATION_SCHEMA_VERSION

    schema_version: Literal["sol_execbench.rdna4_validation.v2"] = (
        RDNA4_VALIDATION_SCHEMA_VERSION
    )
    generated_at: str
    status: Literal["passed", "failed"]
    release_eligible: bool
    source_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    source_dirty: bool
    target: Rdna4TargetSchema
    pytest: Rdna4PytestSummary
    artifacts: list[Rdna4Artifact] = Field(min_length=1)
    attestation: Rdna4Attestation
    payload_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )


@dataclass(frozen=True, slots=True)
class Rdna4EnvironmentIdentity:
    """Exact hardware and user-space identity accepted by the local gate."""

    gfx_target: str
    device_name: str
    device_index: int | None
    rocm_version: str
    torch_version: str
    hip_version: str
    pci_vendor_id: str
    pci_device_id: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible environment identity."""
        return {
            "gfx_target": self.gfx_target,
            "device_name": self.device_name,
            "device_index": self.device_index,
            "rocm_version": self.rocm_version,
            "torch_version": self.torch_version,
            "hip_version": self.hip_version,
            "pci_vendor_id": self.pci_vendor_id,
            "pci_device_id": self.pci_device_id,
        }


def summarize_junit(path: Path) -> dict[str, int]:
    """Return aggregate pytest counts from one JUnit XML artifact."""
    try:
        root = ElementTree.parse(path).getroot()
    except (ElementTree.ParseError, OSError) as exc:
        raise ValueError("RDNA4 validation JUnit artifact is invalid") from exc
    suites = (
        [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    )
    if not suites:
        raise ValueError("RDNA4 validation JUnit artifact has no test suites")
    return {
        field_name: sum(int(suite.get(field_name, "0")) for suite in suites)
        for field_name in ("tests", "failures", "errors", "skipped")
    }


def validate_environment_payload(payload: object) -> Rdna4EnvironmentIdentity:
    """Require the exact locally validated RX 9060 XT ROCm environment."""
    if not isinstance(payload, dict):
        raise ValueError("RDNA4 environment evidence must be a JSON object")
    data = payload.get("data", payload)
    if not isinstance(data, dict) or data.get("status") != "available":
        raise ValueError("RDNA4 environment evidence is unavailable")
    snapshot = data.get("snapshot")
    if not isinstance(snapshot, dict):
        raise ValueError("RDNA4 environment snapshot is missing")
    snapshot_data = cast(dict[str, Any], snapshot)
    gpus = snapshot_data.get("gpus")
    invalid_device = (
        not isinstance(gpus, list)
        or len(gpus) != 1
        or not isinstance(gpus[0], dict)
        or gpus[0].get("gfx_target") != RDNA4_VALIDATION_GFX_TARGET
    )
    if invalid_device:
        raise ValueError(
            "RDNA4 validation requires exactly one RX 9060 XT gfx1200 GPU",
        )
    pytorch = snapshot_data.get("pytorch")
    if not isinstance(pytorch, dict) or pytorch.get("available") is not True:
        raise ValueError(
            "RDNA4 validation requires an available PyTorch ROCm runtime",
        )
    if (
        pytorch.get("device_count") != 1
        or not isinstance(pytorch.get("device_name"), str)
        or not str(pytorch.get("device_name")).strip()
        or pytorch.get("gfx_target") != RDNA4_VALIDATION_GFX_TARGET
    ):
        raise ValueError(
            "RDNA4 PyTorch device identity does not match the GPU probe",
        )
    torch_version = str(pytorch.get("torch_version", ""))
    hip_version = str(pytorch.get("hip_version", ""))
    if torch_version != RDNA4_VALIDATION_TORCH_VERSION:
        raise ValueError(
            "RDNA4 validation PyTorch version is outside the locked scope",
        )
    if hip_version != RDNA4_VALIDATION_HIP_VERSION:
        raise ValueError(
            "RDNA4 validation HIP version is outside the locked scope",
        )
    rocm_version = _rocm_version(snapshot_data)
    if rocm_version != RDNA4_VALIDATION_ROCM_VERSION:
        raise ValueError(
            "RDNA4 validation ROCm version is outside the locked scope",
        )
    _verify_pci_identity(snapshot_data)
    device_index = gpus[0].get("index")
    if device_index is not None and type(device_index) is not int:
        raise ValueError("RDNA4 validation GPU index is invalid")
    return Rdna4EnvironmentIdentity(
        gfx_target=RDNA4_VALIDATION_GFX_TARGET,
        device_name=str(pytorch.get("device_name", "")),
        device_index=device_index,
        rocm_version=rocm_version,
        torch_version=torch_version,
        hip_version=hip_version,
        pci_vendor_id=RDNA4_VALIDATION_PCI_VENDOR_ID,
        pci_device_id=RDNA4_VALIDATION_PCI_DEVICE_ID,
    )


def _verify_pci_identity(snapshot: dict[str, Any]) -> None:
    tools = snapshot.get("tools")
    amd_smi = tools.get("amd-smi") if isinstance(tools, dict) else None
    parsed = amd_smi.get("parsed") if isinstance(amd_smi, dict) else None
    if not isinstance(parsed, dict) or (
        parsed.get("pci_vendor_ids") != [RDNA4_VALIDATION_PCI_VENDOR_ID]
        or parsed.get("pci_device_ids") != [RDNA4_VALIDATION_PCI_DEVICE_ID]
    ):
        raise ValueError(
            "RDNA4 validation PCI identity does not match the RX 9060 XT",
        )


def _rocm_version(snapshot: dict[str, Any]) -> str:
    rocm = snapshot.get("rocm")
    if isinstance(rocm, dict) and isinstance(rocm.get("version"), str):
        return str(rocm["version"])
    versions: set[str] = set()
    tools = snapshot.get("tools")
    if isinstance(tools, dict):
        for value in tools.values():
            if not isinstance(value, dict):
                continue
            match = _ROCM_PATH_VERSION.search(str(value.get("path", "")))
            if match is not None:
                versions.add(match.group(1))
    return versions.pop() if len(versions) == 1 else ""


def build_validation_manifest(
    *,
    directory: Path,
    source_revision: str,
    source_dirty: bool,
    generated_at: str,
    environment: Rdna4EnvironmentIdentity,
    pytest_returncode: int,
    artifact_paths: Sequence[Path],
    attestation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a deterministic manifest over one local hardware-validation run."""
    junit_path = directory / "pytest-rdna4.xml"
    counts = summarize_junit(junit_path)
    passed = (
        pytest_returncode == 0
        and counts["tests"] > 0
        and counts["failures"] == 0
        and counts["errors"] == 0
        and counts["skipped"] == 0
    )
    artifacts = [
        {
            "path": str(path.relative_to(directory)),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(artifact_paths)
    ]
    model = Rdna4ValidationManifest.model_validate(
        {
            "schema_version": RDNA4_VALIDATION_SCHEMA_VERSION,
            "generated_at": generated_at,
            "status": "passed" if passed else "failed",
            # Local evidence is useful for engineering validation but cannot
            # self-upgrade into trusted release authority.
            "release_eligible": False,
            "source_revision": source_revision,
            "source_dirty": source_dirty,
            "target": environment.to_dict(),
            "pytest": {"returncode": pytest_returncode, **counts},
            "artifacts": artifacts,
            "attestation": attestation
            or {"kind": "local_unsigned", "trusted_execution": False},
        },
    )
    manifest = model.model_dump(mode="json", exclude_none=True)
    manifest["payload_sha256"] = stable_json_checksum(manifest)
    return Rdna4ValidationManifest.model_validate(manifest).model_dump(
        mode="json",
    )


def verify_validation_directory(
    directory: Path,
    *,
    expected_source_revision: str | None = None,
    require_release_eligible: bool = False,
) -> dict[str, Any]:
    """Verify manifest identity, artifacts, environment, and test results."""
    manifest_path = directory / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError("RDNA4 validation manifest is invalid") from exc
    model = Rdna4ValidationManifest.model_validate(manifest)
    payload_sha256 = model.payload_sha256
    unsigned = model.model_dump(mode="json", exclude={"payload_sha256"})
    if payload_sha256 != stable_json_checksum(unsigned):
        raise ValueError("RDNA4 validation manifest checksum mismatch")
    manifest = model.model_dump(mode="json")
    _verify_manifest_contract(
        manifest,
        expected_source_revision=expected_source_revision,
        require_release_eligible=require_release_eligible,
    )
    _verify_artifacts(directory, manifest.get("artifacts"))
    environment = json.loads(
        (directory / "environment-doctor.json").read_text(encoding="utf-8"),
    )
    identity = validate_environment_payload(environment)
    if manifest.get("target") != identity.to_dict():
        raise ValueError(
            "RDNA4 validation target does not match environment evidence",
        )
    stored_counts = {
        field_name: int(manifest["pytest"][field_name])
        for field_name in ("tests", "failures", "errors", "skipped")
    }
    if summarize_junit(directory / "pytest-rdna4.xml") != stored_counts:
        raise ValueError("RDNA4 validation JUnit summary mismatch")
    return manifest


def _verify_manifest_contract(
    manifest: dict[str, Any],
    *,
    expected_source_revision: str | None,
    require_release_eligible: bool,
) -> None:
    if manifest.get("schema_version") != RDNA4_VALIDATION_SCHEMA_VERSION:
        raise ValueError("RDNA4 validation schema mismatch")
    if manifest.get("status") != "passed":
        raise ValueError("RDNA4 validation did not pass")
    pytest_summary = manifest.get("pytest")
    if not isinstance(pytest_summary, dict):
        raise ValueError("RDNA4 validation pytest summary is missing")
    if (
        pytest_summary.get("returncode") != 0
        or pytest_summary.get("tests", 0) <= 0
        or any(
            pytest_summary.get(name) != 0
            for name in ("failures", "errors", "skipped")
        )
    ):
        raise ValueError("RDNA4 validation pytest summary is not clean")
    if expected_source_revision is not None and (
        manifest.get("source_revision") != expected_source_revision
    ):
        raise ValueError("RDNA4 validation source revision mismatch")
    if (
        re.fullmatch(r"[0-9a-f]{40}", str(manifest.get("source_revision", "")))
        is None
    ):
        raise ValueError("RDNA4 validation source revision is invalid")
    if not isinstance(manifest.get("source_dirty"), bool):
        raise ValueError("RDNA4 validation source dirty state is invalid")
    attestation = manifest.get("attestation")
    if (
        manifest.get("release_eligible") is not False
        or not isinstance(attestation, dict)
        or attestation.get("kind") not in _LOCAL_ATTESTATION_KINDS
        or attestation.get("trusted_execution") is not False
    ):
        raise ValueError("RDNA4 local validation authority contract is invalid")
    if require_release_eligible:
        raise ValueError("RDNA4 local validation is not release eligible")


def _verify_artifacts(directory: Path, value: object) -> None:
    if not isinstance(value, list) or not value:
        raise ValueError("RDNA4 validation artifacts are missing")
    observed: set[str] = set()
    for raw in value:
        if not isinstance(raw, dict):
            raise ValueError("RDNA4 validation artifact entry is invalid")
        relative = Path(str(raw.get("path", "")))
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or str(relative) in observed
        ):
            raise ValueError("RDNA4 validation artifact path is invalid")
        observed.add(str(relative))
        path = directory / relative
        if (
            not path.is_file()
            or path.stat().st_size != raw.get("size_bytes")
            or sha256_file(path) != raw.get("sha256")
        ):
            raise ValueError("RDNA4 validation artifact identity mismatch")
    if not _REQUIRED_ARTIFACTS.issubset(observed):
        raise ValueError("RDNA4 validation required artifacts are missing")
