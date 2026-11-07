from __future__ import annotations

import pytest
from pydantic import ValidationError

from sol_execbench.cli.protocol import CliSuccessResponse
from sol_execbench.core.control_plane_schema_versions import (
    CLIArtifactKind,
    ExecutionControlSchema,
)


def _valid_response() -> dict[str, object]:
    return {
        "schema_version": ExecutionControlSchema.CLI_PROTOCOL,
        "artifact_kind": CLIArtifactKind.RESPONSE,
        "ok": True,
        "command": "metadata",
        "data": {},
        "artifacts": [],
        "warnings": [],
    }


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload.pop("command"), "command"),
        (lambda payload: payload.update({"unknown": True}), "unknown"),
        (lambda payload: payload.update({"command": []}), "command"),
        (
            lambda payload: payload.update(
                {"schema_version": f"{payload['schema_version']}-obsolete"}
            ),
            "schema_version",
        ),
    ],
    ids=("missing-field", "unknown-field", "wrong-type", "wrong-version"),
)
def test_cli_response_rejects_invalid_raw_payloads(
    mutation,
    message: str,
) -> None:
    payload = _valid_response()
    mutation(payload)

    with pytest.raises((ValueError, ValidationError), match=message):
        CliSuccessResponse.model_validate(payload)
