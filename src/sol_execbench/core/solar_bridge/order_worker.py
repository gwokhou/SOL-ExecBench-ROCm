# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""CPU search worker; candidate verification always calls the pinned original."""

import argparse
from pathlib import Path

from sol_execbench.core.solar_bridge.order import (
    MAX_ORDER_RESPONSE_BYTES,
    OrderWorkerStatus,
    order_execution_identity,
)
from sol_execbench.core.solar_bridge.order_request import (
    load_order_oracle,
    load_order_request,
)
from solar.analysis.order_contracts import (
    MAX_ORDER_MESSAGE_CHARS,
    ORDER_FILENAME,
)
from solar.artifacts import sha256_file, stable_json_checksum
from solar.artifacts.json import write_json_artifact
from solar.order import search_contractions
from solar.schema_versions import SOLAR_ORDER_SCHEMA_VERSION


def main() -> None:
    """Preserve expected input failures without converting invariants to success."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    parser.add_argument("response", type=Path)
    args = parser.parse_args()
    request = load_order_request(args.request)
    payload: dict[str, object] = {
        "schema_version": SOLAR_ORDER_SCHEMA_VERSION,
        "request_sha256": stable_json_checksum(request.model_dump(mode="json")),
    }
    try:
        conversion = load_order_oracle(request)
        search_contractions(
            conversion,
            request.search,
            execution_identity=order_execution_identity(),
        )
        payload.update(
            {
                "status": OrderWorkerStatus.COMPLETED,
                "record_sha256": sha256_file(
                    request.search.job.projection.output_dir / ORDER_FILENAME
                ),
            }
        )
        code = 0
    except (OSError, ValueError) as exc:
        payload.update(
            {
                "status": OrderWorkerStatus.FAILED,
                "message": str(exc)[:MAX_ORDER_MESSAGE_CHARS],
            }
        )
        code = 1
    write_json_artifact(
        args.response, payload, max_bytes=MAX_ORDER_RESPONSE_BYTES
    )
    raise SystemExit(code)


if __name__ == "__main__":
    main()
