# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Isolated worker for untrusted handler candidate generation."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from sol_execbench.core.data.json_utils import load_json_value
from sol_execbench.core.solar_bridge.worker_io import write_worker_response


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    parser.add_argument("response", type=Path)
    args = parser.parse_args()
    request = load_json_value(args.request)
    try:
        from solar.learning import learn_handler_candidate

        sample = yaml.safe_load(Path(request["sample_path"]).read_text()) or {}
        artifact = learn_handler_candidate(
            node_type=str(request["node_type"]),
            sample_node_data=sample,
            output_dir=str(request["output_dir"]),
            model=str(request["model"]),
        )
        response = {"status": "generated", "artifact": artifact}
        exit_code = 0
    except Exception as exc:
        response = {
            "status": "failed",
            "reason_code": "handler_learning_failed",
            "message": str(exc),
        }
        exit_code = 1
    fallback = {
        "status": "failed",
        "reason_code": "worker_response_failed",
        "message": "worker response serialization failed",
    }
    if not write_worker_response(args.response, response, fallback):
        exit_code = 1
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
