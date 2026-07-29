#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Evaluate a frozen gfx1200 diagnostic-model held-out artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sol_execbench.cli.protocol import (
    EXIT_EXECUTION,
    EXIT_RESULT_FAILED,
    CliResult,
    artifact,
    response_failure,
    response_success,
)
from sol_execbench.core.bench.performance_model.acceptance import (
    DiagnosticAcceptanceManifest,
    evaluate_diagnostic_acceptance,
)
from sol_execbench.core.data.json_utils import (
    atomic_write_json_value,
    load_json_file,
)

COMMAND_NAME = "rdna4 diagnostic acceptance"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    """Write a strict aggregate verdict and return nonzero on rejection."""
    arguments = _parse_args()
    manifest = load_json_file(
        DiagnosticAcceptanceManifest,
        arguments.manifest,
    )
    result = evaluate_diagnostic_acceptance(manifest)
    atomic_write_json_value(arguments.output, result.model_dump(mode="json"))
    exit_code = 0 if result.accepted else EXIT_RESULT_FAILED
    response = response_success(
        COMMAND_NAME,
        CliResult(
            data=result.model_dump(mode="json"),
            artifacts=(
                artifact(arguments.output, "diagnostic_acceptance_json"),
            ),
            exit_code=exit_code,
        ),
    )
    print(json.dumps(response, sort_keys=True))
    return exit_code


def _entrypoint() -> int:
    try:
        return main()
    except Exception as error:  # noqa: BLE001 -- standalone JSON boundary
        print(json.dumps(response_failure(COMMAND_NAME, error), sort_keys=True))
        return EXIT_EXECUTION


if __name__ == "__main__":
    raise SystemExit(_entrypoint())
