"""Boundary parsers for baseline command input artifacts."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import click

from sol_execbench.core.scoring.release_baseline import AuthorityInput


def load_json(path: Path, description: str) -> object:
    """Load a JSON artifact and convert parse failures into CLI errors."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise click.ClickException(f"invalid {description} JSON: {exc}") from exc


def load_json_object(path: Path, description: str) -> dict[str, object]:
    """Load a JSON object at the CLI boundary."""
    payload = _string_keyed_mapping(load_json(path, description))
    if payload is None:
        raise click.ClickException(f"{description} must be an object")
    return payload


def suite_workloads_from_json(path: Path) -> list[dict[str, str]]:
    """Read the workload list accepted by frozen-suite commands."""
    payload = load_json(path, "suite manifest")
    payload_mapping = _string_keyed_mapping(payload)
    if payload_mapping is not None:
        payload = payload_mapping.get("workloads")
    if not isinstance(payload, list):
        raise click.ClickException(
            "suite manifest must be a JSON list or object with a workloads list"
        )
    workloads: list[dict[str, str]] = []
    for index, item in enumerate(payload):
        workload = _string_keyed_mapping(item)
        if workload is None:
            raise click.ClickException(f"suite workload {index} must be an object")
        workloads.append(
            {key: value for key, value in workload.items() if isinstance(value, str)}
        )
    return workloads


def authority_from_json(path: Path | None) -> dict[tuple[str, str], AuthorityInput]:
    """Parse optional release authority input without leaking raw JSON inward."""
    if path is None:
        return {}
    payload = load_json(path, "authority")
    payload_mapping = _string_keyed_mapping(payload)
    if payload_mapping is not None:
        payload = payload_mapping.get("workloads")
    if not isinstance(payload, list):
        raise click.ClickException(
            "authority JSON must be a list or object with a workloads list"
        )

    authority: dict[tuple[str, str], AuthorityInput] = {}
    try:
        for index, item in enumerate(payload):
            raw = _string_keyed_mapping(item)
            if raw is None:
                raise ValueError(f"authority workload {index} must be an object")
            definition = raw["definition"]
            workload_uuid = raw["workload_uuid"]
            if not isinstance(definition, str) or not isinstance(workload_uuid, str):
                raise ValueError(
                    f"authority workload {index} requires string definition and workload_uuid"
                )
            authority[(definition, workload_uuid)] = AuthorityInput(
                official_blockers=_string_tuple(raw.get("official_blockers", ())),
                bound_ref=_optional_string(raw.get("bound_ref")),
                bound_sha256=_optional_string(raw.get("bound_sha256")),
                hardware_model_ref=_optional_string(raw.get("hardware_model_ref")),
                hardware_model_sha256=_optional_string(
                    raw.get("hardware_model_sha256")
                ),
            )
    except (KeyError, TypeError, ValueError) as exc:
        raise click.ClickException(f"invalid authority JSON: {exc}") from exc
    return authority


def _string_keyed_mapping(value: object) -> dict[str, object] | None:
    if not isinstance(value, Mapping):
        return None
    return {str(key): item for key, item in value.items()}


def _optional_string(value: object) -> str | None:
    if value is None or isinstance(value, str):
        return value
    raise ValueError("authority reference fields must be strings")


def _string_tuple(value: object) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError("official_blockers must be a list of strings")
    if not all(isinstance(item, str) for item in value):
        raise ValueError("official_blockers must be a list of strings")
    return tuple(item for item in value if isinstance(item, str))


__all__ = [
    "authority_from_json",
    "load_json",
    "load_json_object",
    "suite_workloads_from_json",
]
